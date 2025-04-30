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
})

translations["fr"].update({
    "prompt_sequence_tab": "Séquence de Prompts",
    "prompt_sequence_dict_header": "Gérer le Dictionnaire",
    "prompt_sequence_dict_label": "Dictionnaire (JSON)",
    "prompt_sequence_sequence_label": "Séquence de Prompts",
    "prompt_sequence_run": "Exécuter la Séquence",
    "prompt_sequence_result": "Résultat de la Séquence",
})

class PromptSequenceWidget(Widget):
    def __init__(self, name, prefix, plugin_manager, init_dict=None):
        super().__init__(name, prefix, plugin_manager)
        self.init_dict = init_dict or {}

    def display(self):
        st.header(t("prompt_sequence_dict_header"))
        dict_input = st.text_area(
            t("prompt_sequence_dict_label"),
            value=json.dumps(self.init_dict, indent=2),
            key=f"{self.prefix}_dict_input"
        )
        try:
            work_dict = json.loads(dict_input)
        except json.JSONDecodeError:
            st.error("Invalid JSON format in dictionary")
            work_dict = {}

        st.header(t("prompt_sequence_sequence_label"))
        sequence = st.text_area(
            t("prompt_sequence_sequence_label"),
            height=300,
            key=f"{self.prefix}_sequence_input"
        )

        if st.button(t("prompt_sequence_run"), key=f"{self.prefix}_run_button"):
            result = self.prompt_sequence(sequence, work_dict)
            st.subheader(t("prompt_sequence_result"))
            st.write(result)

    def prompt_sequence(self, sequence: str, work_dict: dict) -> str:
        work_dict = work_dict.copy()
        prompts = sequence.split("#")[1:]  # Skip empty first split
        final_result = ""

        for prompt in prompts:
            lines = prompt.strip().split("\n")
            title = lines[0].strip()
            persona = title.split(":")[1] if ":" in title else None
            sub_prompts = []
            current_sub = ""
            for line in lines[1:]:
                if line.strip() == "---":
                    if current_sub:
                        sub_prompts.append(current_sub.strip())
                        current_sub = ""
                else:
                    current_sub += line + "\n"
            if current_sub:
                sub_prompts.append(current_sub.strip())

            # Replace variables in sub-prompts
            formatted_sub_prompts = []
            for sub in sub_prompts:
                formatted = sub
                for key, value in work_dict.items():
                    formatted = formatted.replace(f"{{{key}}}", str(value))
                formatted_sub_prompts.append(formatted)

            # Execute prompt with LLM
            result = self.process_with_llm(
                formatted_sub_prompts,
                sysprompt=None,
                context=None,
                repeat_on_failure=True,
                number_repeat=3
            )

            # Store result in work_dict
            prompt_name = title.split(":")[0].strip()
            work_dict[prompt_name] = result
            final_result = result

            # Set persona for this prompt if specified
            if persona and persona != "None":
                original_persona = self.plugin_manager.config.get("llm", {}).get("current_persona", "None")
                self.plugin_manager.config.setdefault("llm", {})["current_persona"] = persona
                self.plugin_manager.save_config(self.plugin_manager.config)
                # Restore original persona after execution
                self.plugin_manager.config["llm"]["current_persona"] = original_persona
                self.plugin_manager.save_config(self.plugin_manager.config)

        return final_result
