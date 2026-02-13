from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
import torch
import random
from io import BytesIO
from PIL import Image

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "flux_tab": "Flux Plugin",
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
    # Nouvelles traductions pour le sélecteur de modèle
    "model_choice": "Select Model",
    "model_flux": "FLUX.2 Klein (fast, low VRAM)",
    "model_glm": "GLM-Image (high quality, slower, high VRAM)",
})

translations["fr"].update({
    "flux_tab": "Plugin Flux",
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
    # Nouvelles traductions pour le sélecteur de modèle
    "model_choice": "Choisir le modèle",
    "model_flux": "FLUX.2 Klein (rapide, faible VRAM)",
    "model_glm": "GLM-Image (haute qualité, plus lent, forte VRAM)",
})

class FluxPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)

        self.standard_sizes = {
            "square": (1024, 1024),
            "portrait": (768, 1024),
            "landscape": (1024, 768),
            "youtube": (1280, 720),
            "wide": (1024, 576),
            "tall": (576, 1024),
        }

    def get_config_fields(self):
        return {}

    def get_tabs(self):
        return [{"name": t("flux_tab"), "plugin": "flux"}]

    def run(self, config):
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

        # Sélection de la taille (toujours visible)
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
                    # Chargement des pipelines (cachés séparément)
                    @st.cache_resource
                    def load_flux_pipeline():
                        from diffusers import Flux2KleinPipeline
                        pipe = Flux2KleinPipeline.from_pretrained(
                            "./flux2-klein-4b",  # Dossier local pour FLUX.2 Klein
                            torch_dtype=torch.bfloat16,
                            local_files_only=True
                        )
                        return pipe

                    @st.cache_resource
                    def load_glm_pipeline():
                        from diffusers.pipelines.glm_image import GlmImagePipeline
                        # Pour GLM-Image : téléchargez le modèle dans ./glm-image avec :
                        # export HF_HUB_ENABLE_HF_TRANSFER=0
                        # huggingface-cli download zai-org/GLM-Image --local-dir ./glm-image --resume-download
                        pipe = GlmImagePipeline.from_pretrained(
                            "./glm-image",  # Dossier local pour GLM-Image
                            torch_dtype=torch.bfloat16
                        )
                        return pipe

                    if is_flux:
                        pipe = load_flux_pipeline()
                    else:
                        pipe = load_glm_pipeline()

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    if device == "cuda":
                        if is_flux:
                            pipe.enable_sequential_cpu_offload()
                        else:
                            pipe.enable_model_cpu_offload()  # Recommandé pour GLM-Image (~23GB VRAM nécessaire sinon)
                        torch.cuda.empty_cache()
                    else:
                        st.warning("CUDA non détecté : exécution sur CPU (beaucoup plus lent).")
                        pipe.to("cpu")

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

if __name__ == "__main__":
    st.write("Flux Plugin standalone test")
