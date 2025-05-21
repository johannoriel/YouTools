# widgets/prompt_sequence.py
from lib.global_vars import translations, t, alert
from app import Widget
import streamlit as st
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from widgets.product_matcher import get_sentence_model
import re

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

    def replace_variables(self, text: str, work_dict: dict) -> str:
        """Replace variables in text with values from work_dict."""
        formatted = text
        for key, value in work_dict.items():
            formatted = formatted.replace(f"{{{key}}}", str(value))
        return formatted

    def prompt_sequence(self, sequence: str, work_dict: dict, debug: bool = False) -> str:
        work_dict = work_dict.copy()
        prompts = self._split_prompts(sequence)
        final_result = ""
        debug_expander = None
        if debug:
            debug_expander = self._setup_debug_expander(work_dict, len(prompts))

        for idx, prompt in enumerate(prompts):
            result = self._process_single_prompt(prompt, work_dict, debug, debug_expander, idx, len(prompts))
            prompt_name = prompt.strip().split("\n")[0].split(":")[0].strip()
            work_dict[prompt_name] = result
            final_result = result

        return final_result

    def _split_prompts(self, sequence: str) -> list:
        """Split sequence into individual prompts."""
        prompts = sequence.split("#")
        if not prompts[0].strip().startswith("prompt") and prompts[0].strip():
            prompts = [f"prompt1\n{prompts[0]}"] + prompts[1:]
        else:
            prompts = prompts[1:]
        return prompts

    def _setup_debug_expander(self, work_dict: dict, total_prompts: int):
        """Setup debug expander for logging."""
        debug_expander = st.expander("Debug Information", expanded=False)
        with debug_expander:
            progress_bar = st.progress(0)
            st.write(work_dict)
        return {"expander": debug_expander, "progress_bar": progress_bar, "total_prompts": total_prompts}

    def _parse_prompt_lines(self, lines: list, work_dict: dict) -> tuple:
        """Parse prompt lines into sub-prompts, system prompts, and RAG content."""
        sub_prompts = []
        sys_prompts = []
        rag_content = []
        keywords = []
        rag_params = {"top_k": 10, "threshold": 0.1}
        current_sub = ""
        is_rag = False

        for line in lines:
            line = line.strip()
            if line.startswith("%rag:"):
                is_rag = True
                rag_instruction = line.split("%")[1].split(":", 1)[1].strip()
                # Check for n prefix (0-9)
                param_match = re.match(r"(\d{1,2})\s+(.+)", rag_instruction)
                if param_match:
                    top_k, keywords_str = param_match.groups()
                    rag_params["top_k"] = int(top_k)
                    keywords = self.replace_variables(keywords_str, work_dict).split(",")
                else:
                    keywords = self.replace_variables(rag_instruction, work_dict).split(",")
                continue
            elif line == "%endrag%" and is_rag:
                is_rag = False
                content = "\n".join([self.replace_variables(c, work_dict) for c in rag_content])
                sub_prompts.append(content)
                rag_content = []
                continue

            if is_rag:
                rag_content.append(line)
            else:
                if line == "---":
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

        if is_rag and rag_content:
            content = "\n".join([self.replace_variables(c, work_dict) for c in rag_content])
            sub_prompts.append(content)

        return sub_prompts, sys_prompts, rag_content, keywords, rag_params

    def _process_single_prompt(self, prompt: str, work_dict: dict, debug: bool, debug_expander: dict, idx: int, total_prompts: int) -> str:
        """Process a single prompt."""
        lines = prompt.strip().split("\n")
        title = lines[0].strip()
        persona = title.split(":")[1] if ":" in title else None
        sub_prompts, sys_prompts, rag_content, keywords, rag_params = self._parse_prompt_lines(lines[1:], work_dict)

        # Replace variables in sub-prompts and sys-prompts
        formatted_sub_prompts = [self.replace_variables(sub, work_dict) for sub in sub_prompts]
        formatted_sys_prompts = [self.replace_variables(sys.replace("%system%", ""), work_dict) for sys in sys_prompts]

        # Handle RAG processing
        if rag_content or keywords:
            content = "\n".join([self.replace_variables(c, work_dict) for c in rag_content] or formatted_sub_prompts)
            chunks = self.chunk_content(content, chunk_size=500)
            relevant_content = self.rag_search(chunks, keywords, rag_params)
            formatted_sub_prompts = [relevant_content]

        # Log debug information
        if debug and debug_expander:
            self._log_debug_info(debug_expander, title, idx, formatted_sub_prompts, formatted_sys_prompts, keywords, rag_params)

        # Execute prompt with LLM
        result = self._execute_llm(formatted_sub_prompts, formatted_sys_prompts, persona, debug, debug_expander, idx, total_prompts)

        return result

    def _log_debug_info(self, debug_expander: dict, title: str, idx: int, sub_prompts: list, sys_prompts: list, keywords: list, rag_params: dict):
        """Log debug information to the expander."""
        with debug_expander["expander"]:
            st.write(f"**Prompt {idx + 1}: {title}**")
            if keywords:
                st.write(f"RAG Instruction: Keywords={keywords}, Parameters={rag_params}")
            for i, sub in enumerate(sub_prompts, 1):
                st.write(f"Sub-prompt {i}:\n```\n{sub}\n```")
            for i, sys in enumerate(sys_prompts, 1):
                st.write(f"System prompt {i}:\n```\n{sys}\n```")
            debug_expander["progress_bar"].progress((idx + 1) / debug_expander["total_prompts"])

    def _execute_llm(self, sub_prompts: list, sys_prompts: list, persona: str, debug: bool, debug_expander: dict, idx: int, total_prompts: int) -> str:
        """Execute the LLM with the given prompts and persona."""
        original_persona = self.plugin_manager.config.get("llm", {}).get("current_persona", "None")
        if persona and persona != "None":
            self.plugin_manager.config.setdefault("llm", {})["current_persona"] = persona
            self.plugin_manager.save_config(self.plugin_manager.config)

        result = self.process_with_llm(
            sub_prompts,
            sysprompt=sys_prompts if sys_prompts else None,
            context=None,
            repeat_on_failure=True,
            number_repeat=3
        )

        if persona and persona != "None":
            self.plugin_manager.config["llm"]["current_persona"] = original_persona
            self.plugin_manager.save_config(self.plugin_manager.config)

        if debug and debug_expander:
            with debug_expander["expander"]:
                st.markdown(f"**Response:**\n{result}")
                if idx < total_prompts - 1:
                    st.markdown("---")

        return result

    def chunk_content(self, content: str, chunk_size: int = 500) -> list:
        """Split content into chunks of specified size."""
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunks.append(content[i:i + chunk_size])
        return chunks

    def rag_search(self, chunks: list, keywords: list, rag_params: dict) -> str:
        """Search for relevant content in chunks using sentence transformers."""
        model = get_sentence_model()  # Use cached model from product_matcher
        keyword_query = " ".join(keywords)

        # Encode query and chunks
        query_embedding = model.encode([keyword_query], convert_to_tensor=True)
        chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

        # Compute cosine similarities
        #similarities = util.cos_sim(query_embedding, chunk_embeddings)[0].cpu().numpy()
        #import util fail so fallback :
        cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        similarities = cos_sim(query_embedding, chunk_embeddings).cpu().numpy()

        # Get relevant chunks based on parameters
        top_k = rag_params["top_k"]
        threshold = rag_params["threshold"]

        if top_k == 0:
            # Select all chunks above threshold
            relevant_indices = [i for i, sim in enumerate(similarities) if sim > threshold]
        else:
            # Select top-k chunks above threshold
            k = min(top_k, len(chunks))
            relevant_indices = np.argsort(similarities)[-k:][::-1]
            relevant_indices = [i for i in relevant_indices if similarities[i] > threshold]

        # Collect relevant chunks
        relevant_chunks = [chunks[i] for i in relevant_indices]

        return "\n".join(relevant_chunks) if relevant_chunks else "No relevant content found."
