import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

load_dotenv()


# --------- Validation Error Class ---------
class ValidationError:
    def __init__(self, error_type: str, message: str, details: Dict[str, Any] = None):
        self.error_type = error_type
        self.message = message
        self.details = details or {}

    def to_dict(self):
        return {
            "error_type": self.error_type,
            "message": self.message,
            "details": self.details,
        }


# --------- FAISSSearcher Class (Validation + Search) ---------
class FAISSSearcher:
    def __init__(self, mode="rows"):
        self.mode = mode
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.entries = []
        self.required_columns = [
            "ClientID", "WorkerID", "TaskID", "PriorityLevel",
            "Duration", "AvailableSlots", "RequiredSkills",
            "MaxConcurrent", "Phase", "AttributesJSON"
        ]

        # Load example entries for indexing
        if mode == "headers":
            file = os.path.join("data", "correct_headers.json")
            with open(file) as f:
                self.entries = json.load(f)
        elif mode == "rows":
            file = os.path.join("data", "sample_rows.json")
            with open(file) as f:
                data = json.load(f)
                self.entries_obj = data
                self.entries = [json.dumps(row) for row in data]

        self.embeddings = self.model.encode(self.entries)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))

    def validate_data(self, data: List[Dict[str, Any]]) -> List[ValidationError]:
        errors = []

        # Sets for duplicate checks
        client_ids = set()
        worker_ids = set()
        task_ids = set()

        # For saturation and load checks
        phase_durations = {}
        worker_skills = {}
        required_skills = set()
        corun_groups = {}

        # Validate each row
        for row in data:
            # a. Missing columns
            missing_cols = [col for col in self.required_columns if col not in row]
            if missing_cols:
                errors.append(ValidationError(
                    "missing_columns",
                    f"Missing required columns: {', '.join(missing_cols)}",
                    {"row": row}
                ))
                continue  # Skip further validation for this row

            # b. Duplicate IDs
            if "ClientID" in row:
                if row["ClientID"] in client_ids:
                    errors.append(ValidationError(
                        "duplicate_id",
                        f"Duplicate ClientID: {row['ClientID']}",
                        {"id": row["ClientID"]}
                    ))
                client_ids.add(row["ClientID"])

            if "WorkerID" in row:
                if row["WorkerID"] in worker_ids:
                    errors.append(ValidationError(
                        "duplicate_id",
                        f"Duplicate WorkerID: {row['WorkerID']}",
                        {"id": row["WorkerID"]}
                    ))
                worker_ids.add(row["WorkerID"])

            if "TaskID" in row:
                if row["TaskID"] in task_ids:
                    errors.append(ValidationError(
                        "duplicate_id",
                        f"Duplicate TaskID: {row['TaskID']}",
                        {"id": row["TaskID"]}
                    ))
                task_ids.add(row["TaskID"])

            # c. Malformed lists
            # AvailableSlots check
            try:
                slots = row.get("AvailableSlots", [])
                if isinstance(slots, str):
                    slots = json.loads(slots)
                if not all(isinstance(x, int) for x in slots):
                    errors.append(ValidationError(
                        "malformed_list",
                        "AvailableSlots contains non-integer values",
                        {"row": row}
                    ))
            except Exception:
                errors.append(ValidationError(
                    "malformed_list",
                    "Invalid AvailableSlots format",
                    {"row": row}
                ))

            # d. Out-of-range values
            if "PriorityLevel" in row and (row["PriorityLevel"] < 1 or row["PriorityLevel"] > 5):
                errors.append(ValidationError(
                    "out_of_range",
                    "PriorityLevel must be between 1 and 5",
                    {"value": row["PriorityLevel"]}
                ))

            if "Duration" in row and row["Duration"] < 1:
                errors.append(ValidationError(
                    "out_of_range",
                    "Duration must be at least 1",
                    {"value": row["Duration"]}
                ))

            # e. Broken JSON in AttributesJSON
            try:
                attr_json = row.get("AttributesJSON", {})
                if isinstance(attr_json, str):
                    attr_json = json.loads(attr_json)
                if not isinstance(attr_json, dict):
                    errors.append(ValidationError(
                        "invalid_json",
                        "AttributesJSON must be a valid JSON object",
                        {"row": row}
                    ))
            except Exception:
                errors.append(ValidationError(
                    "invalid_json",
                    "Invalid AttributesJSON format",
                    {"row": row}
                ))

            # f. Track phase durations for saturation checks
            phase = row.get("Phase", None)
            if phase is not None:
                phase_durations[phase] = phase_durations.get(phase, 0) + row.get("Duration", 0)

            # g. Track worker skills & required skills
            if "WorkerID" in row and "Skills" in row:
                skills_list = row["Skills"]
                if isinstance(skills_list, str):
                    try:
                        skills_list = [s.strip() for s in skills_list.split(",")]
                    except Exception:
                        skills_list = []
                worker_skills[row["WorkerID"]] = set(skills_list)

            if "RequiredSkills" in row:
                skills_req = row["RequiredSkills"]
                if isinstance(skills_req, str):
                    try:
                        skills_req = [s.strip() for s in skills_req.split(",")]
                    except Exception:
                        skills_req = []
                for sk in skills_req:
                    required_skills.add(sk)

            # h. CoRunGroups for circular dependency
            if "TaskID" in row and "CoRunGroup" in row:
                corun_groups[row["TaskID"]] = row["CoRunGroup"] if isinstance(row["CoRunGroup"], list) else []

        # i. Unknown references - check RequestedTaskIDs in clients
        all_task_ids = task_ids
        for row in data:
            if "RequestedTaskIDs" in row:
                requested = row["RequestedTaskIDs"]
                if isinstance(requested, str):
                    requested = [x.strip() for x in requested.split(",")]
                unknown = set(requested) - all_task_ids
                if unknown:
                    errors.append(ValidationError(
                        "unknown_reference",
                        f"RequestedTaskIDs refer unknown tasks: {unknown}",
                        {"row": row}
                    ))

        # j. Circular co-run groups detection
        def find_cycle(graph, start, visited=None, path=None):
            if visited is None:
                visited = set()
            if path is None:
                path = []
            visited.add(start)
            path.append(start)
            for nxt in graph.get(start, []):
                if nxt not in visited:
                    if find_cycle(graph, nxt, visited, path):
                        return True
                elif nxt in path:
                    return True
            path.pop()
            return False

        for t_id in corun_groups:
            if find_cycle(corun_groups, t_id):
                errors.append(ValidationError(
                    "circular_dependency",
                    f"Circular co-run dependency detected starting at task {t_id}",
                    {"task": t_id}
                ))

        # k. Worker load vs slots check
        for row in data:
            if "AvailableSlots" in row and "MaxLoadPerPhase" in row:
                slots = row["AvailableSlots"]
                if isinstance(slots, str):
                    try:
                        slots = json.loads(slots)
                    except Exception:
                        slots = []
                if len(slots) < row["MaxLoadPerPhase"]:
                    errors.append(ValidationError(
                        "overloaded_worker",
                        f"Worker {row.get('WorkerID', 'unknown')} has fewer available slots than MaxLoadPerPhase",
                        {"row": row}
                    ))

        # l. Phase slot saturation check
        for phase, dur in phase_durations.items():
            total_slots = 0
            for row in data:
                if row.get("Phase") == phase:
                    slots = row.get("AvailableSlots", [])
                    if isinstance(slots, str):
                        try:
                            slots = json.loads(slots)
                        except Exception:
                            slots = []
                    total_slots += len(slots)
            if dur > total_slots:
                errors.append(ValidationError(
                    "phase_saturation",
                    f"Phase {phase} is oversaturated (Duration {dur} > Slots {total_slots})",
                    {"phase": phase}
                ))

        # m. Skill coverage check
        for skill in required_skills:
            workers_with_skill = sum(1 for skills in worker_skills.values() if skill in skills)
            if workers_with_skill == 0:
                errors.append(ValidationError(
                    "skill_coverage",
                    f"No workers with required skill '{skill}' found",
                    {"skill": skill}
                ))

        # n. MaxConcurrent feasibility
        for row in data:
            req_skills = row.get("RequiredSkills", [])
            if isinstance(req_skills, str):
                req_skills = [s.strip() for s in req_skills.split(",")]
            qualified_workers = sum(
                1 for ws in worker_skills.values() if all(skill in ws for skill in req_skills)
            )
            if row.get("MaxConcurrent", 0) > qualified_workers:
                errors.append(ValidationError(
                    "concurrency_infeasible",
                    f"MaxConcurrent ({row.get('MaxConcurrent')}) exceeds qualified workers ({qualified_workers}) for task {row.get('TaskID')}",
                    {"task": row.get('TaskID')}
                ))

        return errors

    def search(self, query: str, top_k=3):
        query_vec = self.model.encode([query])
        D, I = self.index.search(np.array(query_vec), top_k)
        results = [self.entries[i] for i in I[0]]
        if self.mode == "rows":
            results = [json.loads(r) if isinstance(r, str) else r for r in results]
        return results


# --------- GPTAgent Wrapper ---------
class GPTAgent:
    def __init__(self):
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise ValueError("Missing GITHUB_TOKEN env variable")

        endpoint = os.getenv("GITHUB_AI_ENDPOINT", "https://models.github.ai/inference")
        model = os.getenv("GITHUB_AI_MODEL", "meta/Meta-Llama-3.1-70B-Instruct")

        self.client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(github_token)
        )
        self.model_name = model

    def chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            UserMessage(content=user_prompt)
        ]
    
        response = self.client.complete(
            messages=messages,
            model=self.model_name,
            temperature=0.7,
            top_p=1.0,
            max_tokens=1000
        )
    
        return response.choices[0].message.content




# --------- Core Functionalities ---------

def natural_language_search(gpt_agent: GPTAgent, data: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    # Prompt GPT to filter data based on plain English query
    prompt = f"""
You are a data filter assistant. Given data and a user query, return matching rows only.

Data sample (up to 10 entries):
{json.dumps(data[:10], indent=2)}

User query: "{query}"

Return a JSON array with matching rows only.
"""
    result_str = gpt_agent.chat_completion(
        system_prompt="You are a helpful data filter AI.",
        user_prompt=prompt
    )
    try:
        results = json.loads(result_str)
        return results
    except Exception:
        return []


def natural_language_modify(gpt_agent: GPTAgent, data: List[Dict[str, Any]], command: str) -> Dict[str, Any]:
    # Prompt GPT to suggest modifications based on user command
    prompt = f"""
You are a data modification assistant.

Data sample (up to 10 entries):
{json.dumps(data[:10], indent=2)}

User command: "{command}"

Suggest the changes as a JSON object with keys: "suggested_changes", "updated_data"
Return only JSON.
"""
    result_str = gpt_agent.chat_completion(
        system_prompt="You suggest modifications to the data based on commands.",
        user_prompt=prompt
    )
    try:
        changes = json.loads(result_str)
        return changes
    except Exception:
        return {"suggested_changes": [], "updated_data": []}


def nl_to_rule(
    gpt_agent: GPTAgent,
    user_rule_request: str,
    clients: List[Dict[str, Any]],
    workers: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    prompt = f"""
You are a rules converter.

Clients: {json.dumps(clients[:5], indent=2)}
Workers: {json.dumps(workers[:5], indent=2)}
Tasks: {json.dumps(tasks[:5], indent=2)}

Convert this user request into a structured rule JSON:

User Request: "{user_rule_request}"
"""
    result_str = gpt_agent.chat_completion(
        system_prompt="Convert natural language rules to structured JSON.",
        user_prompt=prompt
    )
    try:
        rule = json.loads(result_str)
        return rule
    except Exception:
        return None


def recommend_rules(
    gpt_agent: GPTAgent,
    clients: List[Dict[str, Any]],
    workers: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    prompt = f"""
You are an AI analyst.

Analyze this data and suggest rules that improve consistency or scheduling:

Clients: {json.dumps(clients[:10], indent=2)}
Workers: {json.dumps(workers[:10], indent=2)}
Tasks: {json.dumps(tasks[:10], indent=2)}

Return JSON list of suggested rule objects.
"""
    result_str = gpt_agent.chat_completion(
        system_prompt="Suggest rules based on data.",
        user_prompt=prompt
    )
    try:
        rules = json.loads(result_str)
        return rules
    except Exception:
        return []


# --------- Main DataManager Class ---------
class DataManager:
    def __init__(self):
        self.gpt_agent = GPTAgent()
        self.validator = FAISSSearcher(mode="rows")

        self.clients: List[Dict[str, Any]] = []
        self.workers: List[Dict[str, Any]] = []
        self.tasks: List[Dict[str, Any]] = []
        self.rules: List[Dict[str, Any]] = []
        self.priorities: Dict[str, float] = {}

    def load_files(self, clients_path, workers_path, tasks_path):
        self.clients = pd.read_csv(clients_path).to_dict(orient="records")
        self.workers = pd.read_csv(workers_path).to_dict(orient="records")
        self.tasks = pd.read_csv(tasks_path).to_dict(orient="records")

    def validate_all(self) -> List[ValidationError]:
        combined = self.clients + self.workers + self.tasks
        return self.validator.validate_data(combined)

    def natural_language_search(self, query: str) -> List[Dict[str, Any]]:
        combined = self.clients + self.workers + self.tasks
        return natural_language_search(self.gpt_agent, combined, query)

    def natural_language_modify(self, command: str) -> Dict[str, Any]:
        combined = self.clients + self.workers + self.tasks
        return natural_language_modify(self.gpt_agent, combined, command)

    def add_rule_from_nl(self, user_rule_request: str) -> Optional[Dict[str, Any]]:
        rule = nl_to_rule(self.gpt_agent, user_rule_request, self.clients, self.workers, self.tasks)
        if rule:
            self.rules.append(rule)
        return rule

    def get_recommended_rules(self) -> List[Dict[str, Any]]:
        return recommend_rules(self.gpt_agent, self.clients, self.workers, self.tasks)

    def export_all(self, output_dir="output") -> str:
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(self.clients).to_csv(os.path.join(output_dir, "clients.csv"), index=False)
        pd.DataFrame(self.workers).to_csv(os.path.join(output_dir, "workers.csv"), index=False)
        pd.DataFrame(self.tasks).to_csv(os.path.join(output_dir, "tasks.csv"), index=False)
        with open(os.path.join(output_dir, "rules.json"), "w") as f:
            json.dump(self.rules, f, indent=2)
        with open(os.path.join(output_dir, "priorities.json"), "w") as f:
            json.dump(self.priorities, f, indent=2)
        return output_dir

    def set_priorities(self, priorities: Dict[str, float]):
        # Expecting keys: PriorityLevel, RequestedTaskIDs, Fairness, LoadLimit
        total = sum(priorities.values())
        if total > 0:
            self.priorities = {k: v / total for k, v in priorities.items()}
        else:
            self.priorities = priorities


# --------- Example usage ---------
if __name__ == "__main__":
    dm = DataManager()
    dm.load_files("clients.csv", "workers.csv", "tasks.csv")
    errors = dm.validate_all()
    print("Validation errors found:", len(errors))
    for e in errors:
        print(e.to_dict())

    # Natural language example
    nl_results = dm.natural_language_search("Find all tasks with Duration > 2")
    print("NL Search Results:", nl_results)

    # Add a rule
    new_rule = dm.add_rule_from_nl("Tasks with PriorityLevel 5 must be assigned first")
    print("New rule added:", new_rule)

    # Export
    output_folder = dm.export_all()
    print(f"Data exported to {output_folder}")
