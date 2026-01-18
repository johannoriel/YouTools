from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
import torch
import random
from io import BytesIO
from PIL import Image  # Ajout pour gérer les images uploadées

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "flux_tab": "Flux Plugin",
    "flux_header": "FLUX.2 Image Generator",
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
    "flux_size_wide": "Wide (1024x576)",
    "flux_size_tall": "Tall (576x1024)",
    "flux_size_custom": "Custom",
})

translations["fr"].update({
    "flux_tab": "Plugin Flux",
    "flux_header": "Générateur d'images FLUX.2",
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
    "flux_size_wide": "Large (1024x576)",
    "flux_size_tall": "Haut (576x1024)",
    "flux_size_custom": "Personnalisée",
})

class FluxPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)

        # Définition des tailles standard
        self.standard_sizes = {
            "square": (1024, 1024),
            "portrait": (768, 1024),
            "landscape": (1024, 768),
            "wide": (1024, 576),
            "tall": (576, 1024),
        }

    def get_config_fields(self):
        """Aucun champ de configuration pour l'instant."""
        return {}

    def get_tabs(self):
        """Définit les onglets du plugin dans l'interface."""
        return [{"name": t("flux_tab"), "plugin": "flux"}]

    def run(self, config):
        """Logique principale du plugin."""
        st.header(t("flux_header"))

        # Prompt texte (obligatoire)
        user_prompt = st.text_area(
            t("flux_input_label"),
            height=120,
            value="A cat holding a sign that says hello world"
        )

        # Upload d'image initiale (optionnel pour img2img)
        init_image_file = st.file_uploader(
            t("flux_init_image_label"),
            type=["png", "jpg", "jpeg", "webp"]
        )

        init_image = None
        target_width = None
        target_height = None

        if init_image_file:
            init_image = Image.open(init_image_file).convert("RGB")
            st.image(init_image, caption="Image initiale", width='stretch')

            # Garder la taille originale pour img2img
            target_width, target_height = init_image.size

        else:
            # Sélection de la taille pour T2I (pas d'image initiale)
            st.subheader(t("flux_size_label"))

            # Options de taille
            size_options = [
                t("flux_size_square"),
                t("flux_size_portrait"),
                t("flux_size_landscape"),
                t("flux_size_wide"),
                t("flux_size_tall"),
                t("flux_size_custom")
            ]

            selected_size_option = st.selectbox(
                "Choisissez une taille d'image",
                size_options,
                index=0
            )

            # Déterminer la taille en fonction de la sélection
            if selected_size_option == t("flux_size_square"):
                target_width, target_height = self.standard_sizes["square"]
            elif selected_size_option == t("flux_size_portrait"):
                target_width, target_height = self.standard_sizes["portrait"]
            elif selected_size_option == t("flux_size_landscape"):
                target_width, target_height = self.standard_sizes["landscape"]
            elif selected_size_option == t("flux_size_wide"):
                target_width, target_height = self.standard_sizes["wide"]
            elif selected_size_option == t("flux_size_tall"):
                target_width, target_height = self.standard_sizes["tall"]
            elif selected_size_option == t("flux_size_custom"):
                # Options personnalisées
                col1, col2 = st.columns(2)
                with col1:
                    target_width = st.number_input(
                        "Largeur",
                        min_value=256,
                        max_value=2048,
                        value=1024,
                        step=64
                    )
                with col2:
                    target_height = st.number_input(
                        "Hauteur",
                        min_value=256,
                        max_value=2048,
                        value=1024,
                        step=64
                    )

            # Afficher la taille sélectionnée
            st.info(f"Taille d'image : {target_width} x {target_height} pixels")

        # Bouton pour lancer la génération
        if st.button(t("flux_process_button")):
            if not user_prompt.strip():
                st.warning("Veuillez saisir un prompt.")
                return

            if not target_width or not target_height:
                st.warning("Veuillez spécifier une taille d'image.")
                return

            with st.spinner(t("flux_processing")):
                try:
                    # Chargement du pipeline mis en cache
                    @st.cache_resource
                    def load_flux_pipeline():
                        from diffusers import Flux2KleinPipeline

                        pipe = Flux2KleinPipeline.from_pretrained(
                            "./flux2-klein-4b",  # Ton dossier local
                            torch_dtype=torch.bfloat16,
                            local_files_only=True
                        )
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        if device == "cuda":
                            pipe.enable_sequential_cpu_offload()  # Garde l'offload agressif pour 8GB VRAM
                            torch.cuda.empty_cache()
                        else:
                            st.warning("CUDA non détecté : exécution sur CPU (beaucoup plus lent).")
                            pipe.to("cpu")
                        return pipe, device

                    pipe, device = load_flux_pipeline()

                    # Seed aléatoire
                    seed = random.randint(0, 999999999)
                    generator = torch.Generator(device=device).manual_seed(seed)

                    # Préparation des paramètres
                    pipe_params = {
                        "prompt": user_prompt,
                        "height": target_height,
                        "width": target_width,
                        "guidance_scale": 1.0,
                        "num_inference_steps": 4,
                        "generator": generator
                    }

                    # Si image initiale → mode img2img
                    if init_image:
                        # Redimensionner l'image initiale à la taille cible (garder les proportions si besoin)
                        # FLUX.2 peut gérer différentes tailles, donc on utilise la taille originale
                        pipe_params["image"] = init_image

                    # Génération
                    image = pipe(**pipe_params).images[0]

                    # Affichage
                    st.image(image, caption=user_prompt, width='stretch')

                    # Téléchargement
                    buf = BytesIO()
                    image.save(buf, format="PNG")
                    buf.seek(0)
                    st.download_button(
                        label="Télécharger l'image",
                        data=buf,
                        file_name=f"flux_{seed}.png",
                        mime="image/png"
                    )

                    st.success(t("flux_success"))

                except Exception as e:
                    st.error(f"Une erreur s'est produite : {str(e)}")
                    st.exception(e)  # Affiche plus de détails pour le débogage


if __name__ == "__main__":
    st.write("Flux Plugin standalone test")
