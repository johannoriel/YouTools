# widgets/prompt_sequence.py
from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import json

translations["en"].update({
    "prompt_sequence_tab": "Prompt Sequence",
    "prompt_sequence_dict_header": "Manage Dictionary",
    "prompt_sequence_dict_label": "Dictionary (JSON)",
    "prompt_sequence_sequence_label": "Prompt Sequence",
    "prompt_sequence_run": "Run Sequence",
    "prompt_sequence_result": "Sequence Result",
    "prompt_sequence_log": "Execution Log",
    "prompt_sequence_prompt": "Prompt {index}",
    "prompt_sequence_response": "Response",
})

translations["fr"].update({
    "prompt_sequence_tab": "Séquence de Prompts",
    "prompt_sequence_dict_header": "Gérer le Dictionnaire",
    "prompt_sequence_dict_label": "Dictionnaire (JSON)",
    "prompt_sequence_sequence_label": "Séquence de Prompts",
    "prompt_sequence_run": "Exécuter la Séquence",
    "prompt_sequence_result": "Résultat de la Séquence",
    "prompt_sequence_log": "Journal d'exécution",
    "prompt_sequence_prompt": "Prompt {index}",
    "prompt_sequence_response": "Réponse",
})

# widgets/prompt_sequence.py
class PromptSequenceWidget(Widget):
    def __init__(self, name, prefix, plugin_manager, init_dict=None):
        super().__init__(name, prefix, plugin_manager)
        self.init_dict = init_dict or {}

    def display(self, input_dict=None):
        st.header(t("prompt_sequence_dict_header"))
        dict_value = input_dict if input_dict is not None else self.init_dict

        # Initialize session state for dictionary entries
        if f"{self.prefix}_dict_entries" not in st.session_state:
            st.session_state[f"{self.prefix}_dict_entries"] = [
                {"id": k, "value": v} for k, v in dict_value.items()
            ]

        # Dictionary editor
        st.write("Dictionary Entries")
        cols = st.columns([2, 3, 1])
        with cols[0]:
            st.write("Identifier")
        with cols[1]:
            st.write("Value")
        with cols[2]:
            st.write("Action")

        # Display and edit existing entries
        for i, entry in enumerate(st.session_state[f"{self.prefix}_dict_entries"]):
            with st.container():
                cols = st.columns([2, 3, 1])
                with cols[0]:
                    entry["id"] = st.text_input(
                        "Identifier",
                        value=entry["id"],
                        key=f"{self.prefix}_dict_id_{i}"
                    )
                with cols[1]:
                    entry["value"] = st.text_area(
                        "Value",
                        value=entry["value"],
                        key=f"{self.prefix}_dict_value_{i}",
                        height=100
                    )
                with cols[2]:
                    if st.button("Delete", key=f"{self.prefix}_delete_{i}"):
                        st.session_state[f"{self.prefix}_dict_entries"].pop(i)
                        st.rerun()

        # Add new entry
        if st.button("Add New Entry", key=f"{self.prefix}_add_entry"):
            st.session_state[f"{self.prefix}_dict_entries"].append({"id": "", "value": ""})
            st.rerun()

        # Convert entries to dictionary
        work_dict = {}
        for entry in st.session_state[f"{self.prefix}_dict_entries"]:
            if entry["id"].strip():
                try:
                    # Try to parse value as JSON to handle potential multiline strings
                    parsed_value = json.loads(entry["value"])
                except json.JSONDecodeError:
                    # If not valid JSON, treat as string
                    parsed_value = entry["value"]
                work_dict[entry["id"]] = parsed_value

        # Display current dictionary as JSON for reference
        st.text_area(
            t("prompt_sequence_dict_label"),
            value=json.dumps(work_dict, indent=2),
            disabled=True,
            key=f"{self.prefix}_dict_output"
        )

        st.header(t("prompt_sequence_sequence_label"))
        sequence = st.text_area(
            t("prompt_sequence_sequence_label"),
            height=300,
            key=f"{self.prefix}_sequence_input"
        )

        if st.button(t("prompt_sequence_run"), key=f"{self.prefix}_run_button"):
            result = self.prompt_sequence(sequence, work_dict, debug=True)
            st.subheader(t("prompt_sequence_result"))
            st.write(result)

    def prompt_sequence(self, sequence: str, work_dict: dict, debug: bool = False) -> str:
        work_dict = work_dict.copy()
        # Split sequence by '#' and handle single prompt case
        prompts = sequence.split("#")
        if not prompts[0].strip().startswith("prompt") and prompts[0].strip():
            prompts = [f"prompt1\n{prompts[0]}"] + prompts[1:]  # Add artificial title
        else:
            prompts = prompts[1:]  # Skip empty first split if it starts with '#'

        final_result = ""
        debug_expander = None
        if debug:
            debug_expander = st.expander("Debug Information", expanded=False)
            with debug_expander:
                progress_bar = st.progress(0)
                total_prompts = len(prompts)
                st.write(work_dict)

        for idx, prompt in enumerate(prompts):
            lines = prompt.strip().split("\n")
            title = lines[0].strip()
            persona = title.split(":")[1] if ":" in title else None
            sub_prompts = []
            sys_prompts = []
            current_sub = ""
            for line in lines[1:]:
                if line.strip() == "---":
                    if current_sub:
                        if "%system%" in current_sub:
                            sys_prompts.append(current_sub.strip().replace("%system%", ""))
                        else:
                            sub_prompts.append(current_sub.strip())
                        current_sub = ""
                else:
                    current_sub += line + "\n"
            if current_sub:
                if "%system%" in current_sub:
                    sys_prompts.append(current_sub.strip().replace("%system%", ""))
                else:
                    sub_prompts.append(current_sub.strip())

            # Replace variables in sub-prompts and sys-prompts
            formatted_sub_prompts = []
            formatted_sys_prompts = []
            for sub in sub_prompts:
                formatted = sub
                for key, value in work_dict.items():
                    formatted = formatted.replace(f"{{{key}}}", str(value))
                formatted_sub_prompts.append(formatted)
            for sys in sys_prompts:
                formatted = sys
                for key, value in work_dict.items():
                    formatted = formatted.replace(f"{{{key}}}", str(value))
                formatted_sys_prompts.append(formatted)

            if debug and debug_expander:
                with debug_expander:
                    st.write(f"**Prompt {idx + 1}: {title}**")
                    for i, sub in enumerate(formatted_sub_prompts, 1):
                        st.write(f"Sub-prompt {i}:\n```\n{sub}\n```")
                    for i, sys in enumerate(formatted_sys_prompts, 1):
                        st.write(f"System prompt {i}:\n```\n{sys}\n```")
                    progress_bar.progress((idx + 1) / total_prompts)

            # Execute prompt with LLM
            original_persona = self.plugin_manager.config.get("llm", {}).get("current_persona", "None")
            if persona and persona != "None":
                self.plugin_manager.config.setdefault("llm", {})["current_persona"] = persona
                self.plugin_manager.save_config(self.plugin_manager.config)

            result = self.process_with_llm(
                formatted_sub_prompts,
                sysprompt=formatted_sys_prompts if formatted_sys_prompts else None,
                context=None,
                repeat_on_failure=True,
                number_repeat=3
            )

            if persona and persona != "None":
                self.plugin_manager.config["llm"]["current_persona"] = original_persona
                self.plugin_manager.save_config(self.plugin_manager.config)

            # Store result in work_dict
            prompt_name = title.split(":")[0].strip()
            work_dict[prompt_name] = result
            final_result = result

            if debug and debug_expander:
                with debug_expander:
                    st.markdown(f"**Response:**\n{result}")
                    if idx < len(prompts) - 1:
                        st.markdown("---")

        return final_result
