import argparse
import torch
import matplotlib.pyplot as plt
import random
from PIL import Image
import os
import streamlit as st
from app import Plugin
from global_vars import t, translations
from diffusers import FluxPipeline, AutoPipelineForImage2Image
from rembg import remove, new_session
import json

# Add translations for this plugin
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
})

class ImggenPlugin(Plugin):
    def __init__(self, name, plugin_manager):
        super().__init__(name, plugin_manager)
        self.pipe = None
        self.prompt_history = []
        self.load_prompt_history()

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
        }

    def get_tabs(self):
        return [{"name": t("generate_image"), "plugin": "imggen"}]

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

    def run(self, config):
        st.header(t("generate_image"))

        # Initialize session state for inputs
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
        with col2:
            if st.button(t("thumbnail_preset")):
                st.session_state.imggen_aspect_ratio = "16:9"
                st.session_state.imggen_remove_background = False
                st.session_state.imggen_use_face = False

        st.subheader(t("prompt_history"))
        selected_history_prompt = st.selectbox("", [""] + self.prompt_history)
        if selected_history_prompt:
            st.session_state.imggen_prompt = selected_history_prompt

        prompt = st.text_area(t("prompt"), key="imggen_prompt", height=150)
        aspect_ratio = st.selectbox(t("aspect_ratio"), ["1:1", "16:9"], key="imggen_aspect_ratio")
        remove_background = st.checkbox(t("remove_background"), key="imggen_remove_background")
        background_removal_method = st.selectbox(t("background_removal_method"), ["ai", "color"], key="imggen_background_removal_method")
        use_random_seed = st.checkbox(t("random_seed"), key="imggen_use_random_seed")
        seed = st.number_input(t("seed"), value=st.session_state.imggen_seed, key="imggen_seed", disabled=use_random_seed)
        use_face = st.checkbox(t("use_face"), key="imggen_use_face")
        steps = st.number_input(t("steps"), min_value=1, value=st.session_state.imggen_steps, key="imggen_steps")
        input_image = st.file_uploader(t("input_image"), type=["png", "jpg", "jpeg"])

        styles = config['imggen']['styles'].split(',')
        style = st.selectbox(t("style"), [""] + [s.strip() for s in styles], key="imggen_style")

        if st.button(t("generate")):
            with st.spinner(t("imggen_processing")):
                if input_image:
                    input_image = Image.open(input_image).convert("RGB")

                self.add_to_prompt_history(prompt)

                sub_prompts = [p.strip() for p in prompt.split('\n') if p.strip()]
                background_prompt = ', '+ config['imggen']['background_prompt']
                self.generate_images( background_prompt,
                    sub_prompts, aspect_ratio, remove_background, background_removal_method,
                    None if use_random_seed else seed, use_face, steps, input_image,
                    config['imggen']['face_prompt'], style, config['imggen']['output_dir']
                )

    def generate_images(self, background_prompt, prompts, aspect_ratio, remove_background, background_removal_method,
                        seed, use_face, steps, input_image, face_prompt, style, output_dir):
        num_columns = 3  # Nombre de colonnes dans la galerie
        cols = st.columns(num_columns)  # Crée les colonnes une fois pour la galerie

        # Barre de progression unique avec un placeholder
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)

        # Génération et affichage progressif des images
        for i, sub_prompt in enumerate(prompts):
            full_prompt = sub_prompt
            if style:
                full_prompt += f", style: {style}"

            # Génère l'image
            image, used_seed = self.generate_image(background_prompt,
                full_prompt, aspect_ratio, remove_background, background_removal_method,
                seed, use_face, steps, input_image, face_prompt
            )

            # Choisir la colonne dans laquelle afficher l'image
            col_idx = i % num_columns
            with cols[col_idx]:  # Mise à jour dans la colonne correspondante
                # Affichage de l'image directement
                st.image(image, caption=f"Image {i+1}/{len(prompts)} \nSeed: {used_seed}\nPrompt: {full_prompt}", use_column_width=True)

            # Sauvegarder l'image
            self.save_image(image, output_dir, sub_prompt)

            # Mettre à jour la barre de progression
            progress_bar.progress((i + 1) / len(prompts))

        # Lorsque tout est terminé, remplacez la barre de progression par un message
        progress_placeholder.empty()  # Efface la barre de progression
        st.success(t("imggen_done"))



    def generate_image(self, background_prompt, prompt, aspect_ratio="1:1", remove_background=True, background_removal_method="ai", seed=None, face=True, steps=2, input_image=None, face_prompt=""):
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        print(f"seed : {seed}")
        generator = torch.Generator().manual_seed(seed)

        original_prompt = prompt
        if face and input_image is None:
            prompt = face_prompt + ", " + prompt

        if remove_background:
            prompt += background_prompt

        if aspect_ratio == "1:1":
            height, width = 1024, 1024
        elif aspect_ratio == "16:9":
            height, width = 1080, 1920
            if face and input_image is None:
                print("Warning: 16:9 aspect ratio is not suitable for face generation. Disabling face mode.")
                face = False
                prompt = original_prompt
        else:
            raise ValueError("Invalid aspect ratio. Choose '1:1' or '16:9'.")

        print(f"Using prompt: {prompt}")

        if self.pipe is None:
            ckpt_id = "black-forest-labs/FLUX.1-schnell"
            if input_image:
                self.pipe = AutoPipelineForImage2Image.from_pretrained(ckpt_id, torch_dtype=torch.bfloat16)
            else:
                self.pipe = FluxPipeline.from_pretrained(ckpt_id, torch_dtype=torch.bfloat16)
            self.pipe.vae.enable_tiling()
            self.pipe.vae.enable_slicing()
            self.pipe.enable_sequential_cpu_offload()

        if input_image:
            image = self.pipe(
                prompt,
                image=input_image,
                num_inference_steps=steps,
                guidance_scale=0.0,
                generator=generator,
            ).images[0]
        else:
            image = self.pipe(
                prompt,
                num_inference_steps=steps,
                guidance_scale=0.0,
                height=height,
                width=width,
                generator=generator,
            ).images[0]

        if remove_background:
            if background_removal_method == "color":
                image = self.remove_green_background_improved(image)
            elif background_removal_method == "ai":
                image = self.remove_background_ai(image, face)

        return image, seed

    @staticmethod
    def remove_green_background_improved(image):
        import cv2
        import numpy as np

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
    def remove_background_ai(image, is_face):
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
    def save_image(image, output_dir, prompt):
        os.makedirs(output_dir, exist_ok=True)
        filename = "_".join(prompt.split()[:5])
        filename = f"{filename}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        print(f"Image saved to: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Generate an image using FLUX pipeline.")
    parser.add_argument("prompt", type=str, help="The prompt for image generation")
    parser.add_argument("-i", "--input-image", type=str, help="Path to input image for img2img")
    parser.add_argument("-ar", "--aspect_ratio", choices=["1:1", "16:9"], default="1:1", help="Aspect ratio of the image")
    parser.add_argument("-nb", "--no-background-removal", action="store_false", dest="remove_background", help="Don't remove the background")
    parser.add_argument("-m", "--method", choices=["color", "ai"], default="ai", help="Method for background removal")
    parser.add_argument("-s", "--seed", type=int, default=3212316546, help="Random seed for generation")
    parser.add_argument("-nf", "--no-face", action="store_false", dest="face", help="Don't add face description to the prompt")
    parser.add_argument("-rs", "--random-seed", action="store_true", help="Use a random seed for generation")
    parser.add_argument("-n", "--steps", type=int, default=2, help="Number of steps for generation")
    parser.add_argument("-o", "--output", type=str, default="~/Images", help="Output directory for saving the image")

    args = parser.parse_args()
    if args.random_seed:
        args.seed = None

    input_image = None
    if args.input_image:
        input_image = Image.open(args.input_image).convert("RGB")

    plugin = ImggenPlugin("imggen", None)
    image, used_seed = plugin.generate_image(", arrière plan blanc vif uni",
        args.prompt, args.aspect_ratio, args.remove_background, args.method,
        args.seed, args.face, args.steps, input_image
    )
    print(f"Image generated with seed: {used_seed}")

    output_dir = os.path.expanduser(args.output)
    plugin.save_image(image, output_dir, args.prompt)

    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
