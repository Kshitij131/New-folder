import streamlit as st
import pandas as pd
import json
from backend import DataManager, ValidationError

st.set_page_config(page_title="Data Alchemist Dashboard", layout="wide")
st.title("🧪 Data Alchemist - AI-Powered Data Manager")

# Instantiate DataManager (singleton in session state)
if "dm" not in st.session_state:
    st.session_state.dm = DataManager()

dm: DataManager = st.session_state.dm


def upload_csv(label: str):
    uploaded_file = st.file_uploader(f"Upload {label} CSV file", type=["csv"], key=label)
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"{label} uploaded with {len(df)} rows")
        return df
    return None


with st.sidebar:
    st.header("1. Upload Data Files")
    clients_df = upload_csv("Clients")
    workers_df = upload_csv("Workers")
    tasks_df = upload_csv("Tasks")

    if st.button("Load Uploaded Files"):
        if clients_df is not None and workers_df is not None and tasks_df is not None:
            dm.clients = clients_df.to_dict(orient="records")
            dm.workers = workers_df.to_dict(orient="records")
            dm.tasks = tasks_df.to_dict(orient="records")
            st.success("All files loaded successfully!")
        else:
            st.error("Please upload all three files before loading.")

    st.markdown("---")
    st.header("2. Set Priority Weights")
    priority_levels = st.slider("PriorityLevel weight", 0.0, 1.0, 0.3)
    requested_tasks = st.slider("RequestedTaskIDs weight", 0.0, 1.0, 0.2)
    fairness = st.slider("Fairness weight", 0.0, 1.0, 0.3)
    load_limit = st.slider("LoadLimit weight", 0.0, 1.0, 0.2)

    if st.button("Set Priorities"):
        priorities = {
            "PriorityLevel": priority_levels,
            "RequestedTaskIDs": requested_tasks,
            "Fairness": fairness,
            "LoadLimit": load_limit
        }
        dm.set_priorities(priorities)
        st.success(f"Priorities set: {priorities}")


# --- Main workspace ---
st.header("3. Data Validation")
if st.button("Run Validation"):
    errors = dm.validate_all()
    if not errors:
        st.success("No validation errors found!")
    else:
        st.error(f"Found {len(errors)} validation errors:")
        for err in errors:
            err_dict = err.to_dict()
            st.write(f"**{err_dict['error_type']}**: {err_dict['message']}")
            st.json(err_dict["details"])

st.markdown("---")
st.header("4. Natural Language Search")
nl_query = st.text_input("Enter a search query (e.g. 'tasks with duration > 3')")

if st.button("Search"):
    if not nl_query.strip():
        st.warning("Please enter a query")
    else:
        results = dm.natural_language_search(nl_query)
        if results:
            st.write(f"Found {len(results)} matching rows:")
            st.json(results)
        else:
            st.info("No results found")

st.markdown("---")
st.header("5. Add New Rule (Natural Language)")
new_rule_input = st.text_area("Describe your new rule to add:")

if st.button("Add Rule"):
    if not new_rule_input.strip():
        st.warning("Please describe the rule")
    else:
        rule = dm.add_rule_from_nl(new_rule_input)
        if rule:
            st.success("Rule added successfully:")
            st.json(rule)
        else:
            st.error("Failed to add rule. Please check the format or try again.")

# New section for AI Recommended Rules
st.markdown("---")
st.header("6. AI Recommended Rules")

if st.button("Get AI Rule Recommendations"):
    with st.spinner("Fetching AI recommendations..."):
        recommended_rules = dm.get_recommended_rules()
        if recommended_rules:
            st.success(f"Found {len(recommended_rules)} recommended rules:")
            for i, rule in enumerate(recommended_rules, 1):
                st.markdown(f"**Rule {i}:**")
                st.json(rule)
        else:
            st.info("No recommended rules found.")

st.markdown("---")
st.header("7. Export Data & Rules")

if st.button("Export All to CSV/JSON"):
    outdir = dm.export_all()
    st.success(f"Exported data and rules to folder: {outdir}")
