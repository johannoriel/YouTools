import os
import sys

# Now import other modules
import argparse
import torch
import matplotlib.pyplot as plt
import random
from PIL import Image
import streamlit as st
from app import Plugin
from lib.global_vars import t, translations
# Note: diffusers imports are delayed until needed
from rembg import remove, new_session
import json
import cv2
import numpy as np
import traceback
from io import BytesIO

# Mise à jour des traductions pour inclure le nouvel onglet
translations["en"].update({
    "generate_image": "Generate Image",
    "prompt": "Prompt",
    "aspect_ratio": "Aspect Ratio",
    "remove_background": "Remove Background",
    "background_removal_method": "Background Removal Method",
    "seed": "Seed",
    "random_seed": "Random Seed",
    "use_face": "Add Face Description",
    "steps": "Number of Steps",
    "input_image": "Input Image",
    "generate": "Generate",
    "image_generated": "Image generated with seed",
    "style": "Style",
    "face_preset": "Face Preset",
    "thumbnail_preset": "Thumbnail Preset",
    "prompt_history": "Prompt History",
    "imggen_processing": "Processing...",
    "imggen_done": "Image generation done !",
    "number_of_images": "Number of Images",
    "backremove_tab": "Background Removal",
    "backremove_header": "Remove Background from Image",
    "upload_image": "Upload Image",
    "process_button": "Remove Background",
    "backremove_processing": "Processing image...",
    "backremove_success": "Background removed successfully!",
    "download_button": "Download Result",
    "prompt_generator_tab": "Prompt Generator",
    "prompt_generator_header": "Generate Image Prompt",
    "user_description": "Describe what you want to draw",
    "generate_prompt": "Generate Prompt",
    "generated_prompt": "Generated Prompt",
    "image_generator_title": "Image Generator",
    # Flux plugin translations
    "flux_tab": "Flux & GLM Generator",
    "flux_header": "FLUX.2 & GLM-Image Generator",
    "flux_input_label": "Enter the prompt for the image you want to generate or edit",
    "flux_init_image_label": "Upload an initial image to edit (optional)",
    "flux_strength_label": "Strength (how much to change the initial image)",
    "flux_strength_help": "0.0 = keep the original image almost unchanged, 1.0 = full generation from prompt",
    "flux_process_button": "Generate Image",
    "flux_processing": "Generating the image, please wait...",
    "flux_success": "Image generated successfully!",
    "flux_size_label": "Image Size",
    "flux_size_square": "Square (1024x1024)",
    "flux_size_portrait": "Portrait (768x1024)",
    "flux_size_landscape": "Landscape (1024x768)",
    "flux_size_youtube": "Youtube (1280x720)",
    "flux_size_wide": "Wide (1024x576)",
    "flux_size_tall": "Tall (576x1024)",
    "flux_size_custom": "Custom",
    "model_choice": "Select Model",
    "model_flux": "FLUX.2 Klein (fast, low VRAM)",
    "model_glm": "GLM-Image (high quality, slower, high VRAM)",
    "composition_tab": "Auto Composition",
    "composition_header": "Automatic Image Composition with Person and Title",
    "composition_title": "Title",
    "composition_description": "Scene Description",
    "composition_person_image": "Person Image",
    "composition_generate": "Generate Composition",
    "composition_processing": "Creating your composition...",
    "composition_success": "Composition created successfully!",
    "composition_person_bg_removed": "Background removed from person image",
    "composition_generating_scene": "Generating the scene...",
    "composition_composing": "Composing final image...",
})

translations["fr"].update({
    "generate_image": "Générer une Image",
    "prompt": "Prompt",
    "aspect_ratio": "Ratio d'Aspect",
    "remove_background": "Supprimer l'Arrière-plan",
    "background_removal_method": "Méthode de Suppression d'Arrière-plan",
    "seed": "Graine",
    "random_seed": "Graine Aléatoire",
    "use_face": "Ajouter une Description de Visage",
    "steps": "Nombre d'Étapes",
    "input_image": "Image d'Entrée",
    "generate": "Générer",
    "image_generated": "Image générée avec la graine",
    "style": "Style",
    "face_preset": "Préréglage Visage",
    "thumbnail_preset": "Préréglage Vignette",
    "prompt_history": "Historique des Prompts",
    "imggen_processing": "En cours...",
    "imggen_done": "Génération d'images terminée !",
    "number_of_images": "Nombre d'Images",
    "backremove_tab": "Suppression d'Arrière-plan",
    "backremove_header": "Supprimer l'arrière-plan d'une image",
    "upload_image": "Télécharger une image",
    "process_button": "Supprimer l'arrière-plan",
    "backremove_processing": "Traitement de l'image...",
    "backremove_success": "Arrière-plan supprimé avec succès !",
    "download_button": "Télécharger le résultat",
    "prompt_generator_tab": "Générateur de Prompt",
    "prompt_generator_header": "Générer un Prompt d'Image",
    "user_description": "Décrivez ce que vous voulez dessiner",
    "generate_prompt": "Générer le Prompt",
    "generated_prompt": "Prompt Généré",
    "image_generator_title": "Générateur d'Image",
    # Flux plugin translations
    "flux_tab": "Générateur Flux & GLM",
    "flux_header": "Générateur d'images FLUX.2 & GLM-Image",
    "flux_input_label": "Entrez le prompt pour l'image que vous souhaitez générer ou modifier",
    "flux_init_image_label": "Téléchargez une image initiale à modifier (optionnel)",
    "flux_strength_label": "Force de modification (combien changer l'image initiale)",
    "flux_strength_help": "0.0 = garder l'image originale presque inchangée, 1.0 = génération complète à partir du prompt",
    "flux_process_button": "Générer l'image",
    "flux_processing": "Génération de l'image en cours, veuillez patienter...",
    "flux_success": "Image générée avec succès !",
    "flux_size_label": "Taille de l'image",
    "flux_size_square": "Carré (1024x1024)",
    "flux_size_portrait": "Portrait (768x1024)",
    "flux_size_landscape": "Paysage (1024x768)",
    "flux_size_youtube": "Youtube (1280x720)",
    "flux_size_wide": "Large (1024x576)",
    "flux_size_tall": "Haut (576x1024)",
    "flux_size_custom": "Personnalisée",
    "model_choice": "Choisir le modèle",
    "model_flux": "FLUX.2 Klein (rapide, faible VRAM)",
    "model_glm": "GLM-Image (haute qualité, plus lent, forte VRAM)",
    "composition_tab": "Composition Auto",
    "composition_header": "Composition automatique d'image avec personne et titre",
    "composition_title": "Titre",
    "composition_description": "Description de la scène",
    "composition_person_image": "Image de la personne",
    "composition_generate": "Générer la composition",
    "composition_processing": "Création de votre composition...",
    "composition_success": "Composition créée avec succès !",
    "composition_person_bg_removed": "Arrière-plan supprimé de l'image de la personne",
    "composition_generating_scene": "Génération de la scène...",
    "composition_composing": "Composition de l'image finale...",
})


class ImggenPlugin(Plugin):
    def __init__(self, name, plugin_manager):
        super().__init__(name, plugin_manager)
        self.pipe = None
        self.prompt_history = []
        self.load_prompt_history()
        self.result_image = None  # Pour stocker l'image sans arrière-plan
        self._diffusers_loaded = False  # Track if diffusers has been loaded

        # Flux plugin attributes
        self.standard_sizes = {
            "square": (1024, 1024),
            "portrait": (768, 1024),
            "landscape": (1024, 768),
            "youtube": (1280, 720),
            "wide": (1024, 576),
            "tall": (576, 1024),
        }
        self.flux_pipe = None
        self.glm_pipe = None

    def _lazy_load_diffusers(self):
        """Lazy load diffusers only when needed with proper CUDA handling"""
        if not self._diffusers_loaded:
            try:
                # Clear any existing CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Try to set CUDA if available
                cuda_available = False
                try:
                    cuda_available = torch.cuda.is_available()
                    if cuda_available:
                        # Test CUDA
                        torch.cuda.init()
                        test_tensor = torch.tensor([1.0], device="cuda")
                        del test_tensor
                        torch.cuda.empty_cache()

                        # Set CUDA visible devices
                        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                        cuda_available = True
                except Exception as e:
                    st.warning(f"CUDA test failed: {str(e)}")
                    os.environ['CUDA_VISIBLE_DEVICES'] = ''
                    cuda_available = False

                # Now import diffusers
                if cuda_available:
                    st.info("Loading diffusers with CUDA support...")
                else:
                    st.info("Loading diffusers with CPU only...")

                from diffusers import Flux2KleinPipeline, AutoPipelineForImage2Image
                from diffusers.pipelines.glm_image import GlmImagePipeline
                self._Flux2KleinPipeline = Flux2KleinPipeline
                self._AutoPipelineForImage2Image = AutoPipelineForImage2Image
                self._GlmImagePipeline = GlmImagePipeline
                self._diffusers_loaded = True

                if cuda_available:
                    st.success("Diffusers loaded with CUDA support")
                else:
                    st.success("Diffusers loaded with CPU only")

            except Exception as e:
                st.error(f"Failed to load diffusers: {str(e)}")
                # Force CPU mode and try again
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                try:
                    from diffusers import Flux2KleinPipeline, AutoPipelineForImage2Image
                    from diffusers.pipelines.glm_image import GlmImagePipeline
                    self._Flux2KleinPipeline = Flux2KleinPipeline
                    self._AutoPipelineForImage2Image = AutoPipelineForImage2Image
                    self._GlmImagePipeline = GlmImagePipeline
                    self._diffusers_loaded = True
                    st.success("Diffusers loaded in CPU mode after fallback")
                except:
                    raise

    def get_config_fields(self):
        return {
            "output_dir": {
                "type": "text",
                "label": t("output_directory"),
                "default": "~/Images"
            },
            "styles": {
                "type": "textarea",
                "label": t("styles"),
                "default": "photorealistic, cartoon, anime, sketch, oil painting, watercolor"
            },
            "face_prompt": {
                "type": "text",
                "label": t("face_prompt"),
                "default": "le visage d'un homme caucasien, brun, yeux bleus, légère barbe"
            },
            "background_prompt": {
                "type": "text",
                "label": t("background_prompt"),
                "default": ", arrière plan blanc vif uni"
            },
            # Flux plugin config fields
            "flux_model_path": {
                "type": "text",
                "label": "FLUX.2 Model Path",
                "default": "./flux2-klein-4b"
            },
            "glm_model_path": {
                "type": "text",
                "label": "GLM-Image Model Path",
                "default": "./glm-image"
            },
        }

    def get_tabs(self):
        return [
            {"name": t("generate_image"), "plugin": "imggen"},
            {"name": t("backremove_tab"), "plugin": "imggen"},
            {"name": t("prompt_generator_tab"), "plugin": "imggen"},
            {"name": t("image_generator_title"), "plugin": "imggen"},
            {"name": t("flux_tab"), "plugin": "imggen"},
            {"name": t("composition_tab"), "plugin": "imggen"}  # Nouveau tab
        ]

    def load_prompt_history(self):
        history_file = 'imggen_prompt_history.json'
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                self.prompt_history = json.load(f)
        else:
            self.prompt_history = []

    def save_prompt_history(self):
        history_file = 'imggen_prompt_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.prompt_history[-20:], f)

    def add_to_prompt_history(self, prompt):
        if prompt in self.prompt_history:
            self.prompt_history.remove(prompt)
        self.prompt_history.insert(0, prompt)
        self.prompt_history = self.prompt_history[:20]
        self.save_prompt_history()

    def get_device(self):
        """Get the appropriate device (cuda/cpu) with fallback"""
        try:
            if torch.cuda.is_available():
                torch.cuda.init()
                test_tensor = torch.tensor([1.0], device="cuda")
                del test_tensor
                torch.cuda.empty_cache()
                return "cuda"
        except:
            pass
        return "cpu"

    def setup_pipeline(self, pipe_type, model_path, is_flux=True):
        """Setup pipeline for flux or glm models"""
        self._lazy_load_diffusers()

        device = self.get_device()
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        try:
            if is_flux:
                pipe = self._Flux2KleinPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    local_files_only=True
                )
            else:
                pipe = self._GlmImagePipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype
                )

            if device == "cuda":
                if is_flux:
                    pipe.enable_sequential_cpu_offload()
                else:
                    pipe.enable_model_cpu_offload()
            else:
                pipe.to("cpu")

            return pipe
        except Exception as e:
            st.error(f"Failed to load pipeline: {str(e)}")
            return None

    # ==================== Original Imggen Methods ====================

    def run_generate_image(self, config):
        st.header(t("generate_image"))

        if 'imggen_prompt' not in st.session_state:
            st.session_state.imggen_prompt = ""
        if 'imggen_aspect_ratio' not in st.session_state:
            st.session_state.imggen_aspect_ratio = "1:1"
        if 'imggen_remove_background' not in st.session_state:
            st.session_state.imggen_remove_background = True
        if 'imggen_background_removal_method' not in st.session_state:
            st.session_state.imggen_background_removal_method = "ai"
        if 'imggen_seed' not in st.session_state:
            st.session_state.imggen_seed = 3212316546
        if 'imggen_use_random_seed' not in st.session_state:
            st.session_state.imggen_use_random_seed = False
        if 'imggen_use_face' not in st.session_state:
            st.session_state.imggen_use_face = True
        if 'imggen_steps' not in st.session_state:
            st.session_state.imggen_steps = 2
        if 'imggen_style' not in st.session_state:
            st.session_state.imggen_style = ""

        col1, col2 = st.columns(2)
        with col1:
            if st.button(t("face_preset")):
                st.session_state.imggen_aspect_ratio = "1:1"
                st.session_state.imggen_remove_background = True
                st.session_state.imggen_use_face = True
                st.session_state.imggen_seed = 3212316546
                st.session_state.imggen_use_random_seed = False
                st.session_state.imggen_style = "oil painting"
        with col2:
            if st.button(t("thumbnail_preset")):
                st.session_state.imggen_aspect_ratio = "16:9"
                st.session_state.imggen_remove_background = False
                st.session_state.imggen_use_face = False
                st.session_state.imggen_use_random_seed = True
                st.session_state.imggen_style = "oil painting"

        aspect_ratio = st.selectbox(t("aspect_ratio"), [
                                    "1:1", "4:3", "3:4", "16:9", "9:16"], key="imggen_aspect_ratio")
        remove_background = st.checkbox(
            t("remove_background"), key="imggen_remove_background")
        background_removal_method = st.selectbox(t("background_removal_method"), [
                                                 "ai", "color"], key="imggen_background_removal_method")
        use_random_seed = st.checkbox(
            t("random_seed"), key="imggen_use_random_seed")
        seed = st.number_input(
            t("seed"), key="imggen_seed", disabled=use_random_seed)
        use_face = st.checkbox(t("use_face"), key="imggen_use_face")
        steps = st.number_input(
            t("steps"), min_value=1, key="imggen_steps")
        input_image = st.file_uploader(
            t("input_image"), type=["png", "jpg", "jpeg"])
        styles = config['imggen']['styles'].split(',')
        style = st.selectbox(
            t("style"), [""] + [s.strip() for s in styles], key="imggen_style")
        multi_styles_input = st.text_input(
            "Multiple Styles (comma-separated, e.g., photorealistic, cartoon, anime)", "", key="imggen_multi_styles")
        if 'imggen_num_images' not in st.session_state:
            st.session_state.imggen_num_images = 1
        num_images = st.number_input(t("number_of_images"), min_value=1,
                                     value=st.session_state.imggen_num_images, key="imggen_num_images")
        manual_seeds_input = st.text_input(
            "Manual Seeds (comma-separated, e.g., 123, 456, 789)", "", key="imggen_manual_seeds")

        st.subheader(t("prompt_history"))
        selected_history_prompt = st.selectbox(t("prompt_history"), [""] + self.prompt_history, label_visibility="collapsed")
        if selected_history_prompt:
            st.session_state.imggen_prompt = selected_history_prompt
        prompt = st.text_area(t("prompt"), key="imggen_prompt", height=150)

        if st.button(t("generate")):
            with st.spinner(t("imggen_processing")):
                if input_image:
                    input_image = Image.open(input_image).convert("RGB")
                self.add_to_prompt_history(prompt)
                sub_prompts = [p.strip()
                               for p in prompt.split('\n') if p.strip()]
                background_prompt = ', ' + \
                    config['imggen']['background_prompt']
                manual_seeds = None
                if manual_seeds_input:
                    try:
                        manual_seeds = [int(s.strip())
                                        for s in manual_seeds_input.split(',')]
                    except ValueError:
                        st.error(
                            "Invalid seed list format. Please use comma-separated integers (e.g., 123, 456, 789)")
                        return
                multi_styles = [s.strip() for s in multi_styles_input.split(
                    ',')] if multi_styles_input else None
                self.generate_images(background_prompt, sub_prompts, aspect_ratio, remove_background, background_removal_method,
                                     None if use_random_seed or num_images > 1 else seed, use_face, steps, input_image,
                                     config['imggen']['face_prompt'], style, config['imggen']['output_dir'],
                                     num_images, manual_seeds, multi_styles)

    def generate_images(self, background_prompt, prompts, aspect_ratio, remove_background, background_removal_method,
                        seed, use_face, steps, input_image, face_prompt, style, output_dir, num_images, manual_seeds=None, multi_styles=None):
        num_columns = 3
        cols = st.columns(num_columns)
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)

        if manual_seeds:
            seeds = manual_seeds
        else:
            seeds = [seed if seed is not None else random.randint(
                0, 2**32 - 1) for _ in range(num_images)]

        total_images = len(seeds) * len(prompts) * \
            (len(multi_styles) if multi_styles else 1)
        image_count = 0

        for i, current_seed in enumerate(seeds):
            generator = torch.Generator().manual_seed(int(current_seed))
            for j, sub_prompt in enumerate(prompts):
                if multi_styles:
                    styles_to_use = multi_styles
                else:
                    styles_to_use = [style] if style else [""]
                for k, current_style in enumerate(styles_to_use):
                    full_prompt = sub_prompt
                    style_suffix = f", style: {current_style}" if current_style else ""
                    full_prompt += style_suffix
                    image, _ = self.generate_image(background_prompt, full_prompt, aspect_ratio, remove_background,
                                                   background_removal_method, current_seed, use_face, steps, input_image, face_prompt)
                    col_idx = image_count % num_columns
                    with cols[col_idx]:
                        caption = f"Image {i+1}/{len(seeds)} \nSeed: {current_seed}\nPrompt: {sub_prompt}"
                        if current_style:
                            caption += f"\nStyle: {current_style}"
                        st.image(image, caption=caption,
                                 width='stretch')
                    self.save_image(image, output_dir, sub_prompt,
                                    current_seed, current_style)
                    image_count += 1
                    progress_bar.progress(image_count / total_images)
        progress_placeholder.empty()
        st.success(t("imggen_done"))

    def generate_image(self, background_prompt, prompt, aspect_ratio="1:1", remove_background=True, background_removal_method="ai", seed=None, face=True, steps=2, input_image=None, face_prompt=""):
        # Lazy load diffusers if not already loaded
        self._lazy_load_diffusers()

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator().manual_seed(int(seed))
        original_prompt = prompt
        if face and input_image is None:
            prompt = face_prompt + ", " + prompt
        if remove_background:
            prompt += background_prompt
        if aspect_ratio == "1:1":
            height, width = 1024, 1024
        elif aspect_ratio == "4:3":
            height, width = 768, 1024
        elif aspect_ratio == "3:4":
            height, width = 1024, 768
        elif aspect_ratio == "16:9":
            height, width = 1080, 1920
        elif aspect_ratio == "9:16":
            height, width = 1920, 1080
        else:
            raise ValueError("Invalid aspect ratio.")

        # Check CUDA availability with fallback
        device = "cpu"
        torch_dtype = torch.float32

        try:
            # First check if CUDA is available
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                # Try to initialize CUDA
                try:
                    torch.cuda.init()
                    # Test with a small tensor
                    test_tensor = torch.tensor([1.0], device="cuda")
                    del test_tensor
                    torch.cuda.empty_cache()
                    device = "cuda"
                    torch_dtype = torch.bfloat16
                    st.info("CUDA initialized successfully")
                except Exception as e:
                    st.warning(f"CUDA initialization failed: {str(e)}. Using CPU.")
                    device = "cpu"
                    torch_dtype = torch.float32
        except Exception as e:
            st.warning(f"CUDA check failed: {str(e)}. Using CPU.")
            device = "cpu"
            torch_dtype = torch.float32

        try:
            if self.pipe is None:
                ckpt_id = "black-forest-labs/FLUX.2-klein-4B"
                st.info(f"Loading pipeline for {device} with dtype {torch_dtype}...")

                # Force CPU mode for loading to avoid CUDA issues
                original_env = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                os.environ['CUDA_VISIBLE_DEVICES'] = ''

                try:
                    if input_image:
                        self.pipe = self._AutoPipelineForImage2Image.from_pretrained(
                            ckpt_id, torch_dtype=torch_dtype)
                    else:
                        self.pipe = self._Flux2KleinPipeline.from_pretrained(
                            ckpt_id, torch_dtype=torch_dtype)
                finally:
                    # Restore original env
                    if original_env:
                        os.environ['CUDA_VISIBLE_DEVICES'] = original_env

                # Move to device
                self.pipe = self.pipe.to(device)

                # Apply CPU offload if on CUDA
                if device == "cuda":
                    try:
                        self.pipe.enable_model_cpu_offload()
                    except:
                        pass

                st.success("Pipeline loaded successfully.")

        except RuntimeError as load_error:
            if "CUDA" in str(load_error):
                st.warning("CUDA error during pipeline load, retrying on CPU...")
                device = "cpu"
                torch_dtype = torch.float32
                self.pipe = None
                return self.generate_image(background_prompt, prompt, aspect_ratio, remove_background,
                                           background_removal_method, seed, face, steps, input_image, face_prompt)
            else:
                st.error(f"Failed to load pipeline: {str(load_error)}")
                return None, seed

        try:
            st.info(f"Generating image on {device}...")
            if input_image:
                image = self.pipe(prompt, image=input_image, num_inference_steps=steps,
                                  guidance_scale=0.0, generator=generator).images[0]
            else:
                image = self.pipe(prompt, num_inference_steps=steps, guidance_scale=0.0,
                                  height=height, width=width, generator=generator).images[0]
            st.success("Image generated successfully.")

        except RuntimeError as infer_error:
            if "CUDA" in str(infer_error) and device == "cuda":
                st.warning("CUDA error during inference, retrying on CPU...")
                device = "cpu"
                torch_dtype = torch.float32
                self.pipe = self.pipe.to("cpu")
                generator = torch.Generator().manual_seed(int(seed))

                if input_image:
                    image = self.pipe(prompt, image=input_image, num_inference_steps=steps,
                                      guidance_scale=0.0, generator=generator).images[0]
                else:
                    image = self.pipe(prompt, num_inference_steps=steps, guidance_scale=0.0,
                                      height=height, width=width, generator=generator).images[0]
            else:
                st.error(f"Inference failed: {str(infer_error)}")
                return None, seed

        if remove_background:
            if background_removal_method == "color":
                image = self.remove_green_background_improved(image)
            elif background_removal_method == "ai":
                image = self.remove_background_ai(image, face)
        return image, seed

    @staticmethod
    def remove_green_background_improved(image):
        np_image = np.array(image)
        hsv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2HSV)
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        inverse_mask = cv2.bitwise_not(green_mask)
        alpha = inverse_mask
        rgba_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2RGBA)
        rgba_image[:, :, 3] = alpha
        return Image.fromarray(rgba_image)

    @staticmethod
    def remove_background_ai(image, is_face=False):
        model = "u2net_human_seg" if is_face else "u2net"
        session = new_session(model)
        result = remove(image, session=session)
        if result.mode != 'RGBA':
            result = result.convert('RGBA')
        data = result.getdata()
        new_data = []
        for item in data:
            if item[0] > 250 and item[1] > 250 and item[2] > 250:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
        result.putdata(new_data)
        return result

    @staticmethod
    def save_image(image, output_dir, prompt, seed, style=None):
        import re
        os.makedirs(output_dir, exist_ok=True)
        filename_base = "_".join(prompt.split()[:5])
        style_part = f"_{style.replace(' ', '_')}" if style else ""
        filename = re.sub(r'[^a-zA-Z0-9_]', '',
                          f"{filename_base}{style_part}_{seed}") + ".png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        return filepath

    def run_background_removal(self, config):
        st.header(t("backremove_header"))

        # Upload de l'image
        input_image = st.file_uploader(t("upload_image"), type=[
                                       "png", "jpg", "jpeg"], key="backremove_input")

        # Sélection de la méthode de suppression
        removal_method = st.selectbox(t("background_removal_method"), [
                                      "ai", "color"], key="backremove_method")

        # Bouton pour lancer le traitement
        if st.button(t("process_button")) and input_image:
            with st.spinner(t("backremove_processing")):
                try:
                    # Charger l'image
                    image = Image.open(input_image).convert("RGB")

                    # Supprimer l'arrière-plan
                    if removal_method == "ai":
                        self.result_image = self.remove_background_ai(image)
                    else:  # color
                        self.result_image = self.remove_green_background_improved(
                            image)

                    # Afficher le résultat
                    st.image(self.result_image, caption="Result",
                             width='stretch')
                    st.success(t("backremove_success"))

                    # Sauvegarde et option de téléchargement
                    output_dir = os.path.expanduser(
                        config['imggen']['output_dir'])
                    output_path = self.save_backremove_image(
                        self.result_image, output_dir, input_image.name)

                    with open(output_path, "rb") as file:
                        st.download_button(
                            label=t("download_button"),
                            data=file,
                            file_name=f"no_bg_{input_image.name}",
                            mime="image/png"
                        )

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    @staticmethod
    def save_backremove_image(image, output_dir, original_filename):
        os.makedirs(output_dir, exist_ok=True)
        filename = f"no_bg_{os.path.splitext(original_filename)[0]}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        return filepath

    def run_prompt_generator(self, config):
        st.header(t("prompt_generator_header"))

        # Champ pour la description de l'utilisateur
        user_description = st.text_area(t("user_description"), height=150, key="imggen_user_description")

        # Bouton pour générer le prompt
        if st.button(t("generate_prompt")) and user_description:
            with st.spinner(t("imggen_processing")):
                try:
                    # Système prompt pour guider le LLM
                    sys_prompt = (
                        "You are an expert in generating detailed and vivid prompts for image generation models. "
                        "Based on the user's description, create a concise, descriptive, and creative prompt optimized for an image generation model. "
                        "Include specific details about style, colors, lighting, and composition where relevant. "
                        "Return only the generated prompt without additional explanations."
                        "Respond in english"
                    )

                    # Appeler le LLM pour générer le prompt
                    generated_prompt = self.process_with_llm(user_description, sysprompt=sys_prompt)

                    # Afficher le prompt généré
                    st.text_area(t("generated_prompt"), value=generated_prompt, height=150, key="imggen_generated_prompt")

                    # Option pour copier le prompt dans l'onglet de génération d'image
                    if st.button("Use in Image Generation"):
                        st.session_state.imggen_prompt = st.session_state.imggen_generated_prompt

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    def run_image_generator_widget(self, config):
        # Lazy import pour éviter les problèmes d'importation diffusers
        try:
            from widgets.image_generator import ImageGeneratorWidget
            widget = ImageGeneratorWidget("image_generator", "imggen_widget", plugin_manager=self.plugin_manager)
            widget.display()
        except Exception as e:
            st.error(f"Failed to load Image Generator widget: {str(e)}")
            st.info("Make sure the widget file exists in the widgets directory.")

    # ==================== Flux Plugin Methods ====================

    def run_flux_tab(self, config):
        """Run the Flux/GLM image generation tab"""
        st.header(t("flux_header"))

        # Sélecteur de modèle
        selected_model = st.selectbox(
            t("model_choice"),
            [t("model_flux"), t("model_glm")]
        )
        is_flux = selected_model == t("model_flux")

        # Prompt texte
        user_prompt = st.text_area(
            t("flux_input_label"),
            height=120,
            value="A cat holding a sign that says hello world"
        )

        # Upload image initiale (optionnel)
        init_image_file = st.file_uploader(
            t("flux_init_image_label"),
            type=["png", "jpg", "jpeg", "webp"]
        )

        uploaded_image = None
        if init_image_file:
            uploaded_image = Image.open(init_image_file).convert("RGB")
            st.image(uploaded_image, caption="Image initiale uploadée", use_container_width=True)

        # Sélection de la taille
        st.subheader(t("flux_size_label"))

        size_options = [
            t("flux_size_square"),
            t("flux_size_portrait"),
            t("flux_size_landscape"),
            t("flux_size_youtube"),
            t("flux_size_wide"),
            t("flux_size_tall"),
            t("flux_size_custom")
        ]

        selected_size_option = st.selectbox(
            "Choisissez une taille d'image",
            size_options,
            index=0
        )

        target_width, target_height = 1024, 1024  # Default values

        if selected_size_option == t("flux_size_square"):
            target_width, target_height = self.standard_sizes["square"]
        elif selected_size_option == t("flux_size_portrait"):
            target_width, target_height = self.standard_sizes["portrait"]
        elif selected_size_option == t("flux_size_landscape"):
            target_width, target_height = self.standard_sizes["landscape"]
        elif selected_size_option == t("flux_size_youtube"):
            target_width, target_height = self.standard_sizes["youtube"]
        elif selected_size_option == t("flux_size_wide"):
            target_width, target_height = self.standard_sizes["wide"]
        elif selected_size_option == t("flux_size_tall"):
            target_width, target_height = self.standard_sizes["tall"]
        elif selected_size_option == t("flux_size_custom"):
            col1, col2 = st.columns(2)
            with col1:
                target_width = st.number_input("Largeur", min_value=256, max_value=2048, value=1024, step=64)
            with col2:
                target_height = st.number_input("Hauteur", min_value=256, max_value=2048, value=1024, step=64)

        # Ajustement automatique des dimensions pour GLM-Image (doivent être divisibles par 32)
        if not is_flux:
            original_w, original_h = target_width, target_height
            target_width = max(32, (target_width // 32) * 32)
            target_height = max(32, (target_height // 32) * 32)
            if (target_width, target_height) != (original_w, original_h):
                st.info(f"Dimensions ajustées à {target_width}x{target_height} pour compatibilité GLM-Image (divisible par 32).")

        st.info(f"Taille cible : {target_width} x {target_height} pixels")

        # Préparation de l'image initiale (redimensionnée à la taille cible si présente)
        init_image = None
        strength = None
        if uploaded_image:
            init_image = uploaded_image.resize((target_width, target_height))
            st.image(init_image, caption=f"Image initiale redimensionnée à {target_width}x{target_height}", use_container_width=True)

            # Strength uniquement pour FLUX (GLM-Image n'a pas ce paramètre)
            if is_flux:
                strength = st.slider(
                    t("flux_strength_label"),
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    step=0.05,
                    help=t("flux_strength_help")
                )

        # Bouton génération
        if st.button(t("flux_process_button")):
            if not user_prompt.strip():
                st.warning("Veuillez saisir un prompt.")
                return

            with st.spinner(t("flux_processing")):
                try:
                    # Charger le pipeline approprié
                    model_path = config['imggen']['flux_model_path'] if is_flux else config['imggen']['glm_model_path']

                    if is_flux:
                        if self.flux_pipe is None:
                            self.flux_pipe = self.setup_pipeline("flux", model_path, is_flux=True)
                        pipe = self.flux_pipe
                    else:
                        if self.glm_pipe is None:
                            self.glm_pipe = self.setup_pipeline("glm", model_path, is_flux=False)
                        pipe = self.glm_pipe

                    if pipe is None:
                        st.error("Failed to load the model pipeline.")
                        return

                    device = self.get_device()

                    # Paramètres adaptés au modèle
                    guidance_scale = 1.0 if is_flux else 1.5
                    num_steps = 4 if is_flux else 50

                    seed = random.randint(0, 999999999)
                    generator = torch.Generator(device=device).manual_seed(seed)

                    pipe_params = {
                        "prompt": user_prompt,
                        "height": target_height,
                        "width": target_width,
                        "guidance_scale": guidance_scale,
                        "num_inference_steps": num_steps,
                        "generator": generator
                    }

                    if init_image:
                        if is_flux:
                            pipe_params["image"] = init_image
                            if strength is not None:
                                pipe_params["strength"] = strength
                        else:
                            pipe_params["image"] = [init_image]  # GLM-Image attend une liste

                    image = pipe(**pipe_params).images[0]

                    st.image(image, caption=user_prompt, use_container_width=True)

                    buf = BytesIO()
                    image.save(buf, format="PNG")
                    buf.seek(0)
                    st.download_button(
                        label="Télécharger l'image",
                        data=buf,
                        file_name=f"{'flux' if is_flux else 'glm'}_{seed}.png",
                        mime="image/png"
                    )

                    st.success(t("flux_success"))

                except Exception as e:
                    st.error(f"Une erreur s'est produite : {str(e)}")
                    st.exception(e)

    def run_composition_tab(self, config):
        """Run the automatic composition tab with person and title"""
        st.header(t("composition_header"))

        # Initialisation des variables de session (sauf pour le file_uploader)
        if 'composition_title' not in st.session_state:
            st.session_state.composition_title = ""
        if 'composition_description' not in st.session_state:
            st.session_state.composition_description = ""
        # Ne pas initialiser composition_person_image dans st.session_state

        # Interface utilisateur
        col1, col2 = st.columns(2)

        with col1:
            # Titre
            title = st.text_input(
                t("composition_title"),
                key="composition_title",
                placeholder="Entrez le titre à afficher sur l'image"
            )

            # Description de la scène
            description = st.text_area(
                t("composition_description"),
                key="composition_description",
                height=150,
                placeholder="Décrivez la scène de fond (ex: une plage tropicale au coucher du soleil, avec des palmiers)"
            )

        with col2:
            # Image de la personne - sans key pour éviter l'erreur
            person_image_file = st.file_uploader(
                t("composition_person_image"),
                type=["png", "jpg", "jpeg"]
            )

            if person_image_file:
                person_image = Image.open(person_image_file).convert("RGB")
                st.image(person_image, caption="Image originale", use_container_width=True)

        # Bouton de génération
        if st.button(t("composition_generate")):
            if not title or not description or not person_image_file:
                st.warning("Veuillez remplir tous les champs (titre, description et image de la personne).")
                return

            with st.status(t("composition_processing"), expanded=True) as status:
                try:
                    # Étape 1: Charger l'image de la personne
                    person_image = Image.open(person_image_file).convert("RGB")

                    # Étape 2: Supprimer l'arrière-plan de la personne
                    status.update(label=t("composition_person_bg_removed"))
                    person_no_bg = self.remove_background_ai(person_image, is_face=True)

                    # Redimensionner la personne pour qu'elle tienne dans la composition
                    person_no_bg = self.resize_person_for_composition(person_no_bg)

                    # Afficher l'image sans fond (optionnel)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(person_no_bg, caption="Personne sans fond", use_container_width=True)

                    # Étape 3: Générer la scène de fond avec Flux
                    status.update(label=t("composition_generating_scene"))

                    # Construction du prompt pour la scène
                    scene_prompt = f"Create a beautiful background scene: {description}. High quality, detailed, empty space on the left for a person and on the right for text. The scene should be wide and panoramic."

                    # Taille de l'image finale
                    final_width = 1536
                    final_height = 1024

                    # Générer la scène de fond
                    background_scene = self.generate_composition_background(scene_prompt, final_width, final_height)

                    with col2:
                        st.image(background_scene, caption="Fond généré", use_container_width=True)

                    # Étape 4: Composer l'image finale
                    status.update(label=t("composition_composing"))
                    final_image = self.compose_final_image(
                        background_scene,
                        person_no_bg,
                        title,
                        final_width,
                        final_height
                    )

                    # Afficher le résultat final
                    st.image(final_image, caption="Composition finale", use_container_width=True)

                    # Bouton de téléchargement
                    buf = BytesIO()
                    final_image.save(buf, format="PNG")
                    buf.seek(0)
                    st.download_button(
                        label="Télécharger la composition",
                        data=buf,
                        file_name=f"composition_{title[:30].replace(' ', '_')}.png",
                        mime="image/png"
                    )

                    status.update(label=t("composition_success"), state="complete")

                except Exception as e:
                    st.error(f"Une erreur s'est produite : {str(e)}")
                    st.exception(e)

    def resize_person_for_composition(self, person_image, target_height=800):
        """Redimensionne l'image de la personne pour la composition"""
        # Calculer le ratio pour que la personne fasse environ target_height de haut
        aspect_ratio = person_image.width / person_image.height
        new_width = int(target_height * aspect_ratio)

        # Limiter la largeur pour ne pas prendre trop de place
        if new_width > 500:
            new_width = 500
            new_height = int(500 / aspect_ratio)
        else:
            new_height = target_height

        return person_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def generate_composition_background(self, prompt, width, height):
        """Génère le fond de la composition avec Flux"""
        self._lazy_load_diffusers()

        device = self.get_device()
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        try:
            # Charger le pipeline Flux si pas déjà fait
            if self.flux_pipe is None:
                model_path = self.plugin_manager.config['imggen']['flux_model_path']
                self.flux_pipe = self.setup_pipeline("flux", model_path, is_flux=True)

            if self.flux_pipe is None:
                raise Exception("Impossible de charger le modèle Flux")

            # Générer l'image
            seed = random.randint(0, 999999999)
            generator = torch.Generator(device=device).manual_seed(seed)

            with torch.no_grad():
                image = self.flux_pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    guidance_scale=1.0,
                    num_inference_steps=4,
                    generator=generator
                ).images[0]

            return image

        except Exception as e:
            st.error(f"Erreur lors de la génération du fond : {str(e)}")
            # Créer un fond de secours (dégradé)
            return self.create_fallback_background(width, height)

    def create_fallback_background(self, width, height):
        """Crée un fond de secours si la génération échoue"""
        import numpy as np

        # Créer un dégradé bleu
        array = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            value = int(100 + 155 * y / height)
            array[y, :, 0] = value  # R
            array[y, :, 1] = value  # G
            array[y, :, 2] = 255    # B

        return Image.fromarray(array)

    def compose_final_image(self, background, person, title, width, height):
        """Compose l'image finale avec la personne, le titre et le fond"""
        from PIL import ImageDraw, ImageFont

        # Convertir le fond en RGBA pour la transparence
        background = background.convert("RGBA")

        # Créer un calque pour la composition
        composition = background.copy()

        # Positionner la personne à gauche
        # Laisser une marge de 50px
        person_x = 50
        person_y = (height - person.height) // 2  # Centrer verticalement

        # Superposer la personne
        if person.mode == 'RGBA':
            # Créer un masque pour la transparence
            mask = person.split()[3]
            composition.paste(person, (person_x, person_y), mask)
        else:
            composition.paste(person, (person_x, person_y))

        # Ajouter le titre à droite
        draw = ImageDraw.Draw(composition)

        # Essayer de charger une police, utiliser la police par défaut si pas disponible
        try:
            # Taille de police adaptée à la largeur disponible
            font_size = min(80, int(height / 10))
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        # Zone de texte à droite
        text_x = person.width + 150  # Après la personne + marge
        text_width = width - text_x - 100  # Largeur disponible pour le texte

        # Diviser le titre en lignes si nécessaire
        words = title.split()
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            # Utiliser une estimation approximative de la largeur
            if font.getbbox(test_line)[2] < text_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        # Dessiner chaque ligne
        line_height = font_size + 10
        start_y = (height - (len(lines) * line_height)) // 2

        # Ajouter un contour blanc pour meilleure lisibilité
        for i, line in enumerate(lines):
            y = start_y + i * line_height

            # Dessiner le contour (ombre)
            for offset_x, offset_y in [(-2,-2), (-2,2), (2,-2), (2,2), (0,-2), (0,2), (-2,0), (2,0)]:
                draw.text((text_x + offset_x, y + offset_y), line, font=font, fill=(0, 0, 0, 255))

            # Dessiner le texte principal en blanc
            draw.text((text_x, y), line, font=font, fill=(255, 255, 255, 255))

        return composition

    def run(self, config):
        # Mise à jour des tabs pour inclure le nouveau
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            t("generate_image"),
            t("backremove_tab"),
            t("prompt_generator_tab"),
            t("image_generator_title"),
            t("flux_tab"),
            t("composition_tab")  # Nouveau tab
        ])

        with tab1:
            self.run_generate_image(config)
        with tab2:
            self.run_background_removal(config)
        with tab3:
            self.run_prompt_generator(config)
        with tab4:
            self.run_image_generator_widget(config)
        with tab5:
            self.run_flux_tab(config)
        with tab6:
            self.run_composition_tab(config)


def main():
    parser = argparse.ArgumentParser(
        description="Generate an image using FLUX pipeline.")
    parser.add_argument("prompt", type=str,
                        help="The prompt for image generation")
    parser.add_argument("-i", "--input-image", type=str,
                        help="Path to input image for img2img")
    parser.add_argument("-ar", "--aspect_ratio", choices=[
                        "1:1", "4:3", "3:4", "16:9", "9:16"], default="1:1", help="Aspect ratio of the image")
    parser.add_argument("-nb", "--no-background-removal", action="store_false",
                        dest="remove_background", help="Don't remove the background")
    parser.add_argument(
        "-m", "--method", choices=["color", "ai"], default="ai", help="Method for background removal")
    parser.add_argument("-s", "--seed", type=int,
                        default=3212316546, help="Random seed for generation")
    parser.add_argument("-nf", "--no-face", action="store_false",
                        dest="face", help="Don't add face description to the prompt")
    parser.add_argument("-rs", "--random-seed", action="store_true",
                        help="Use a random seed for generation")
    parser.add_argument("-n", "--steps", type=int, default=2,
                        help="Number of steps for generation")
    parser.add_argument("-o", "--output", type=str, default="~/Images",
                        help="Output directory for saving the image")
    args = parser.parse_args()
    if args.random_seed:
        args.seed = None
    input_image = None
    if args.input_image:
        input_image = Image.open(args.input_image).convert("RGB")
    plugin = ImggenPlugin("imggen", None)
    image, used_seed = plugin.generate_image(", arrière plan blanc vif uni", args.prompt, args.aspect_ratio, args.remove_background, args.method,
                                             args.seed, args.face, args.steps, input_image)
    print(f"Image generated with seed: {used_seed}")
    output_dir = os.path.expanduser(args.output)
    plugin.save_image(image, output_dir, args.prompt)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
