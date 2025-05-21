from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import random
import os
from PIL import Image
from diffusers import FluxPipeline
import torch

translations["en"].update({
    "image_generator_title": "Image Generator",
    "pre_prompt": "Pre-Prompt",
    "dimensions": "Dimensions",
    "generate_image": "Generate Image",
    "generated_prompt": "Generated Prompt",
})

translations["fr"].update({
    "image_generator_title": "Générateur d'Image",
    "pre_prompt": "Pré-Prompt",
    "dimensions": "Dimensions",
    "generate_image": "Générer l'Image",
    "generated_prompt": "Prompt Généré",
})

class ImageGeneratorWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self.pipe = None

    def display(self, pre_prompt=None, dimension=None):
        st.title(t("image_generator_title"))

        # Champs pour le pré-prompt et la dimension
        default_pre_prompt = pre_prompt if pre_prompt else ""
        pre_prompt_input = st.text_area(t("pre_prompt"), value=default_pre_prompt, key=f"{self.prefix}_pre_prompt", height=100)

        default_dimension = dimension if dimension else "1:1"
        dimension_input = st.selectbox(t("dimensions"), ["1:1", "16:9"], index=["1:1", "16:9"].index(default_dimension), key=f"{self.prefix}_dimension")

        # Bouton pour générer l'image
        if st.button(t("generate_image"), key=f"{self.prefix}_generate"):
            with st.spinner("Processing..."):
                image_filename = self.generate_image(pre_prompt_input, dimension_input)
                if image_filename:
                    st.image(image_filename, caption="Generated Image", use_container_width=True)

    def generate_image_direct(self, prompt, dimension, output_path, overwrite=True):
        # Initialisation du pipeline si nécessaire
        if self.pipe is None:
            ckpt_id = "black-forest-labs/FLUX.1-schnell"
            self.pipe = FluxPipeline.from_pretrained(ckpt_id, torch_dtype=torch.bfloat16)
            self.pipe.vae.enable_tiling()
            self.pipe.vae.enable_slicing()
            self.pipe.enable_sequential_cpu_offload()

        # Configuration des dimensions
        if dimension == "1:1":
            height, width = 1024, 1024
        elif dimension == "16:9":
            height, width = 1080, 1920
        else:
            raise ValueError("Invalid dimension.")

        # Génération de l'image avec graine aléatoire
        seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator().manual_seed(int(seed))
        image = self.pipe(
            prompt,
            negative_prompt="text, words, phrase, ugly, messy, blurry, low quality",
            num_inference_steps=2,
            guidance_scale=0.0,
            height=height,
            width=width,
            generator=generator
        ).images[0]

        # Vérifier si output_path est un fichier ou un répertoire
        if os.path.splitext(output_path)[1]:  # Si output_path a une extension (c'est un fichier)
            filepath = output_path
            output_dir = os.path.dirname(output_path) or "."  # Dossier parent, ou "." si aucun
        else:  # Si output_path est un répertoire
            output_dir = output_path
            filename = f"generated_image_{seed}.png"
            filepath = os.path.join(output_dir, filename)

        # Créer le répertoire si nécessaire
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Vérification si le fichier existe et gestion de l'écrasement
        if not overwrite and os.path.exists(filepath):
            raise FileExistsError(f"File already exists: {filepath}")
        image.save(filepath)

        return filepath

    def generate_image(self, pre_prompt, dimension):
        # Génération du prompt à partir du pré-prompt
        sys_prompt = (
            "You are an expert in generating detailed and vivid prompts for image generation models. "
            "Based on the user's description, create a concise, descriptive, and creative prompt optimized for an image generation model. "
            "Include specific details about style, colors, lighting, and composition where relevant. "
            "Return only the generated prompt without additional explanations."
        )
        generated_prompt = self.process_with_llm(pre_prompt, sysprompt=sys_prompt)

        # Affichage discret du prompt généré pour debug
        st.write(t("generated_prompt") + ": " + generated_prompt)

        # Appel à generate_image_direct pour générer l'image dans self.work_dir()
        return self.generate_image_direct(generated_prompt, dimension, self.work_dir(), overwrite=True)
