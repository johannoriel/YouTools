import os
import torch
import streamlit as st
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download
from PIL import Image
from app import Plugin
from lib.global_vars import t, translations

# Add translations for this plugin
translations["en"].update({
    "videogen_tab_text2video": "Text to Video",
    "videogen_tab_image2video": "Image to Video",
    "videogen_tab_video2video": "Video to Video",
    "videogen_header": "Video Generation",
    "videogen_pre_prompt": "Pre-Prompt (Optional)",
    "videogen_generate_prompt": "Generate Prompt",
    "videogen_prompt": "Prompt",
    "videogen_negative_prompt": "Negative Prompt",
    "videogen_fps": "FPS",
    "videogen_quality": "Quality",
    "videogen_seed": "Seed",
    "videogen_steps": "Default Number of Inference Steps",
    "videogen_random_seed": "Random Seed",
    "videogen_generate_video": "Generate Video",
    "videogen_processing": "Generating video...",
    "videogen_done": "Video generation complete!",
    "videogen_input_image": "Select Input Image",
    "videogen_input_video": "Select Input Video",
    "videogen_output_dir": "Output Directory",
})

translations["fr"].update({
    "videogen_tab_text2video": "Texte vers Vidéo",
    "videogen_tab_image2video": "Image vers Vidéo",
    "videogen_tab_video2video": "Vidéo vers Vidéo",
    "videogen_header": "Génération de Vidéo",
    "videogen_pre_prompt": "Pré-Prompt (Optionnel)",
    "videogen_generate_prompt": "Générer le Prompt",
    "videogen_prompt": "Prompt",
    "videogen_negative_prompt": "Prompt Négatif",
    "videogen_fps": "IPS",
    "videogen_quality": "Qualité",
    "videogen_seed": "Graine",
    "video_gen_steps": "Nombre de passes d'inférences",
    "videogen_random_seed": "Graine Aléatoire",
    "videogen_generate_video": "Générer la Vidéo",
    "videogen_processing": "Génération de la vidéo...",
    "videogen_done": "Génération de la vidéo terminée !",
    "videogen_input_image": "Sélectionner une Image d'Entrée",
    "videogen_input_video": "Sélectionner une Vidéo d'Entrée",
    "videogen_output_dir": "Répertoire de Sortie",
})


class VideogenPlugin(Plugin):
    def __init__(self, name, plugin_manager):
        super().__init__(name, plugin_manager)
        self.pipe = None
        self._initialize_session_state()

    def _initialize_session_state(self):
        if 'videogen_prompt' not in st.session_state:
            st.session_state.videogen_prompt = ""
        if 'videogen_negative_prompt' not in st.session_state:
            st.session_state.videogen_negative_prompt = ""
        if 'videogen_fps' not in st.session_state:
            st.session_state.videogen_fps = 15
        if 'videogen_quality' not in st.session_state:
            st.session_state.videogen_quality = 5
        if 'videogen_seed' not in st.session_state:
            st.session_state.videogen_seed = 0
        if 'videogen_use_random_seed' not in st.session_state:
            st.session_state.videogen_use_random_seed = False

    def get_config_fields(self):
        return {
            "model_dir": {
                "type": "text",
                "label": "Model Directory",
                "default": "models/Wan-AI/Wan2.1-T2V-1.3B"
            },
            "i2v_model_dir": {
                "type": "text",
                "label": "I2V Model Directory",
                "default": "models/Wan-AI/Wan2.1-I2V-14B-480P"
            },
            "output_dir": {
                "type": "text",
                "label": t("videogen_output_dir"),
                "default": "~/Videos"
            },
            "default_fps": {
                "type": "number",
                "label": t("videogen_fps"),
                "default": 15
            },
            "default_quality": {
                "type": "number",
                "label": t("videogen_quality"),
                "default": 5
            },
            "default_seed": {
                "type": "number",
                "label": t("videogen_seed"),
                "default": 0
            },
            "default_num_inference_steps": {
                "type": "number",
                "label": t("video_gen_steps"),
                "default": 50
            },
            "default_prompt": {
                "type": "textarea",
                "label": "Default Prompt",
                "default": "A documentary-style photography scene featuring a lively, joyful small dog running swiftly across a lush green lawn. The dog has a tan coat, both ears perked up, and an expression of focus and delight. Sunlight bathes its fur, giving it a soft, shiny look. The background shows a wide expanse of grass dotted with a few wildflowers, with a blue sky and scattered white clouds on the horizon. The perspective is dynamic, capturing the dog's motion mid-run and the vibrancy of the surrounding grass. Side view, medium shot."
            },
            "default_negative_prompt": {
                "type": "textarea",
                "label": "Default Negative Prompt",
                "default": "Overly vibrant colors, overexposure, static image, blurry details, subtitles, artistic style, painted artwork, still image, overall grayish tone, poor or low quality, JPEG compression artifacts, unsightly or incomplete elements, poorly drawn fingers or hands, poorly rendered faces, deformed or distorted shapes, fused fingers, motionless image, cluttered background, three-legged dog, overcrowded background, walking backward."
            },
            "pre_prompt": {
                "type": "textarea",
                "label": "Default Pre-Prompt for Prompt Generation",
                "default": "Generate a detailed prompt and negative prompt in English for video generation based on this description: {input}. The prompt should describe a dynamic, documentary-style scene with vivid details. The negative prompt should exclude unwanted artifacts and styles. Format the output as: '[Prompt]: <prompt text> || [Negative Prompt]: <negative prompt text>'"
            }
        }

    def get_tabs(self):
        return [
            {"name": t("videogen_tab_text2video"),
             "plugin": "videogen_text2video"},
            {"name": t("videogen_tab_image2video"),
             "plugin": "videogen_image2video"},
            {"name": t("videogen_tab_video2video"),
             "plugin": "videogen_video2video"}
        ]

    def _load_pipeline(self, config, is_image_to_video=False):
        if self.pipe is None:
            if is_image_to_video:
                model_dir = config['videogen']['i2v_model_dir']
                if not os.path.exists(model_dir):
                    snapshot_download(
                        "Wan-AI/Wan2.1-I2V-14B-480P", cache_dir="models")
                model_manager = ModelManager(device="cpu")
                model_manager.load_models(
                    [
                        [f"{model_dir}/diffusion_pytorch_model-0000{i}-of-00007.safetensors" for i in range(
                            1, 8)],
                        f"{model_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                        f"{model_dir}/models_t5_umt5-xxl-enc-bf16.pth",
                        f"{model_dir}/Wan2.1_VAE.pth",
                    ],
                    torch_dtype=torch.bfloat16,
                )
            else:
                model_dir = config['videogen']['model_dir']
                if not os.path.exists(model_dir):
                    snapshot_download(
                        "Wan-AI/Wan2.1-T2V-1.3B", cache_dir="models")
                model_manager = ModelManager(device="cpu")
                model_manager.load_models(
                    [
                        f"{model_dir}/diffusion_pytorch_model.safetensors",
                        f"{model_dir}/models_t5_umt5-xxl-enc-bf16.pth",
                        f"{model_dir}/Wan2.1_VAE.pth",
                    ],
                    torch_dtype=torch.bfloat16,
                )
            self.pipe = WanVideoPipeline.from_model_manager(
                model_manager, torch_dtype=torch.bfloat16, device="cuda"
            )
            self.pipe.enable_vram_management(num_persistent_param_in_dit=None)

    def _generate_prompts(self, config, pre_prompt_input):
        pre_prompt = config['videogen']['pre_prompt'].format(
            input=pre_prompt_input)
        response = self.process_with_llm(
            pre_prompt,
            config.get('llm', {}).get('llm_sys_prompt', ''),
            pre_prompt_input
        )
        # Parse the response based on the specified format
        try:
            prompt_part, neg_prompt_part = response.split("||")
            prompt = prompt_part.split("[Prompt]:")[1].strip()
            negative_prompt = neg_prompt_part.split(
                "[Negative Prompt]:")[1].strip()
            return prompt, negative_prompt
        except (ValueError, IndexError):
            # Fallback si le format n'est pas respecté
            return response.strip(), config['videogen']['default_negative_prompt']

    def _generate_video(self, config, prompt, negative_prompt, fps, quality, seed, num_inference_steps, input_image=None, input_video=None):
        # Charger I2V si input_image, T2V sinon
        self._load_pipeline(config, is_image_to_video=bool(input_image))
        output_dir = os.path.expanduser(config['videogen']['output_dir'])
        os.makedirs(output_dir, exist_ok=True)

        video_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "tiled": True
        }
        if input_image:
            video_params["input_image"] = input_image
        elif input_video:
            video_params["input_video"] = input_video
            video_params["denoising_strength"] = 0.7

        video = self.pipe(**video_params)

        import re
        clean_prompt = re.sub(
            r'[^a-zA-Z0-9]', '', prompt.encode('ascii', 'ignore').decode('ascii'))
        truncated_prompt = clean_prompt[:20]
        output_filename = f"video_{truncated_prompt}_{seed}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        save_video(video, output_path, fps=fps, quality=quality)
        return output_path

    def run(self, config):
        tab1, tab2, tab3 = st.tabs([
            t("videogen_tab_text2video"),
            t("videogen_tab_image2video"),
            t("videogen_tab_video2video")
        ])

        with tab1:
            self._text_to_video(config)
        with tab2:
            self._image_to_video(config)
        with tab3:
            self._video_to_video(config)

    def _common_ui(self, config, input_image=None, input_video=None, suffix=""):
        pre_prompt = st.text_area(
            t("videogen_pre_prompt"), height=100, key=f"pre_prompt_{suffix}")
        if st.button(t("videogen_generate_prompt"), key=f"generate_prompt_{suffix}") and pre_prompt:
            prompt, negative_prompt = self._generate_prompts(
                config, pre_prompt)
            st.session_state.videogen_prompt = prompt
            st.session_state.videogen_negative_prompt = negative_prompt
        else:
            st.session_state.videogen_prompt = st.session_state.videogen_prompt or config[
                'videogen']['default_prompt']
            st.session_state.videogen_negative_prompt = st.session_state.videogen_negative_prompt or config[
                'videogen']['default_negative_prompt']

        prompt = st.text_area(
            t("videogen_prompt"), st.session_state.videogen_prompt, height=150, key=f"prompt_{suffix}")
        negative_prompt = st.text_area(
            t("videogen_negative_prompt"), st.session_state.videogen_negative_prompt, height=150, key=f"negative_prompt_{suffix}")
        fps = st.number_input(t("videogen_fps"), min_value=1, value=int(
            config['videogen']['default_fps']), key=f"fps_{suffix}")
        quality = st.number_input(t("videogen_quality"), min_value=1, max_value=10, value=int(
            config['videogen']['default_quality']), key=f"quality_{suffix}")
        num_inference_steps = st.number_input("Number of Inference Steps", min_value=1, value=int(
            config['videogen']['default_num_inference_steps']), key=f"num_inference_steps_{suffix}")
        use_random_seed = st.checkbox(
            t("videogen_random_seed"), key=f"random_seed_{suffix}")
        seed = st.number_input(t("videogen_seed"), value=int(
            config['videogen']['default_seed']), disabled=use_random_seed, key=f"seed_{suffix}")
        seed = None if use_random_seed else seed

        if st.button(t("videogen_generate_video"), key=f"generate_video_{suffix}"):
            with st.spinner(t("videogen_processing")):
                output_path = self._generate_video(
                    config, prompt, negative_prompt, fps, quality, seed, num_inference_steps, input_image, input_video)
                st.video(output_path)
                st.success(t("videogen_done"))
                st.write(f"Video saved at: {output_path}")

    def _text_to_video(self, config):
        st.header(t("videogen_header"))
        self._common_ui(config, suffix="txt2vid")

    def _image_to_video(self, config):
        from modelscope import dataset_snapshot_download
        st.header(t("videogen_header"))
        work_dir = config['common']['work_directory']
        image_files = [f for f in os.listdir(
            work_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        selected_image = st.selectbox(
            t("videogen_input_image"), [""] + image_files)

        if selected_image:  # Check if an image is selected
            input_image = Image.open(os.path.join(work_dir, selected_image))
            self._common_ui(config, input_image=input_image, suffix="img2vid")
        else:
            st.warning("Please select an image to proceed.")

    def _video_to_video(self, config):
        st.header(t("videogen_header"))
        work_dir = config['common']['work_directory']
        video_files = [f for f in os.listdir(work_dir) if f.endswith('.mp4')]
        selected_video = st.selectbox(
            t("videogen_input_video"), [""] + video_files)
        input_video = VideoData(os.path.join(
            work_dir, selected_video), height=480, width=832) if selected_video else None
        self._common_ui(config, input_video=input_video, suffix="vid2vid")


def main():
    # Example usage outside Streamlit for testing
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description="Generate a video using Wan2.1-T2V-1.3B.")
    parser.add_argument("--prompt", type=str,
                        default="A small dog running on a lawn")
    args = parser.parse_args()

    config = {
        "videogen": {
            "model_dir": "models/Wan-AI/Wan2.1-T2V-1.3B",
            "output_dir": "~/Videos",
            "default_fps": 15,
            "default_quality": 5,
            "default_seed": 0,
            "default_prompt": "A documentary-style photography scene...",
            "default_negative_prompt": "Overly vibrant colors...",
            "pre_prompt": "Generate a detailed prompt and negative prompt..."
        },
        "common": {"work_directory": "."}
    }
    plugin = VideogenPlugin("videogen", None)
    plugin._generate_video(
        config, args.prompt, config['videogen']['default_negative_prompt'], 15, 5, 0)


if __name__ == "__main__":
    main()
