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
st.header("5. Add Custom Rules and Apply")

# Tab for different rule operations
tab1, tab2, tab3 = st.tabs(["Add Single Rule", "AI Recommendations", "Apply All Rules"])

with tab1:
    st.subheader("Add and Apply Individual Rule")
    new_rule_input = st.text_area("Describe your rule:", 
                                  placeholder="e.g., High priority clients should be processed first")
    
    if st.button("Add Rule and Apply to Data", type="primary"):
        if new_rule_input.strip():
            with st.spinner("Creating rule and applying to data..."):
                result = dm.add_rule_and_apply(new_rule_input.strip())
                
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    st.success("✅ Rule created and applied!")
                    st.info(f"📁 Modified files saved to: {result['output_directory']}")
                    
                    if result["changes_made"]:
                        st.subheader("Changes Made:")
                        for change in result["changes_made"]:
                            st.write(f"• {change}")
                    
                    with st.expander("View Created Rule"):
                        st.json(result["rule_created"])

with tab2:
    st.subheader("AI Rule Recommendations")
    
    if st.button("Get and Apply AI Recommendations"):
        with st.spinner("Getting AI recommendations and applying..."):
            result = dm.get_ai_recommendations_and_apply()
            
            if "message" in result and "No AI recommendations" in result["message"]:
                st.info("No AI recommendations available")
            else:
                st.success(f"✅ Applied {result['ai_rules_applied']} AI recommendations!")
                st.info(f"📁 Modified files saved to: {result['output_directory']}")
                
                with st.expander("View Applied Rules"):
                    for rule in result["recommended_rules"]:
                        st.json(rule)
                
                with st.expander("View Application Summary"):
                    st.json(result["application_result"]["summary"])

with tab3:
    st.subheader("Apply All Existing Rules")
    
    if dm.rules:
        st.write(f"You have {len(dm.rules)} rules ready to apply:")
        for i, rule in enumerate(dm.rules):
            st.write(f"{i+1}. **{rule.get('name', 'Unnamed')}** ({'Active' if rule.get('isActive') else 'Inactive'})")
    
    if st.button("Apply All Rules to Data"):
        with st.spinner("Applying all rules to data..."):
            result = dm.apply_rules_and_regenerate_files()
            
            if result["files_modified"]:
                st.success("✅ All rules applied!")
                st.info(f"📁 Modified files saved to: {result['output_directory']}")
                st.write(f"Applied {result['applied_rules']} rules successfully")
                
                if result["rule_violations"]:
                    st.warning(f"⚠️ {len(result['rule_violations'])} rule violations found")
            else:
                st.info("No rules to apply")

st.markdown("---")
st.header("7. Export Data & Rules")

if st.button("Export All to CSV/JSON"):
    outdir = dm.export_all()
    st.success(f"Exported data and rules to folder: {outdir}")
