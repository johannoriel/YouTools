from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
import torch
import random
from io import BytesIO

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "flux_tab": "Flux Plugin",
    "flux_header": "FLUX.2 Image Generator",
    "flux_input_label": "Enter the prompt for the image you want to generate",
    "flux_process_button": "Generate Image",
    "flux_processing": "Generating the image, please wait...",
    "flux_success": "Image generated successfully!",
})

translations["fr"].update({
    "flux_tab": "Plugin Flux",
    "flux_header": "Générateur d'images FLUX.2",
    "flux_input_label": "Entrez le prompt pour l'image que vous souhaitez générer",
    "flux_process_button": "Générer l'image",
    "flux_processing": "Génération de l'image en cours, veuillez patienter...",
    "flux_success": "Image générée avec succès !",
})

class FluxPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)

    def get_config_fields(self):
        """Aucun champ de configuration pour l'instant (simple)."""
        return {}

    def get_tabs(self):
        """Définit les onglets du plugin dans l'interface."""
        return [{"name": t("flux_tab"), "plugin": "flux"}]

    def run(self, config):
        """Logique principale du plugin."""

        import torch
        print(torch.__version__)
        print(torch.version.cuda)
        print(torch.cuda.is_available())
        print(torch.cuda.device_count())
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))

        st.header(t("flux_header"))

        # Zone de texte pour le prompt (plus adapté que text_input pour des prompts longs)
        user_prompt = st.text_area(
            t("flux_input_label"),
            height=120,
            value="A cat holding a sign that says hello world"
        )

        # Bouton pour lancer la génération
        if st.button(t("flux_process_button")):
            if not user_prompt.strip():
                st.warning("Veuillez saisir un prompt.")
                return

            with st.spinner(t("flux_processing")):
                try:
                    # Chargement du pipeline mis en cache pour éviter de recharger à chaque fois
                    @st.cache_resource
                    def load_flux_pipeline():
                        from diffusers import Flux2KleinPipeline

                        pipe = Flux2KleinPipeline.from_pretrained(
                            "./flux2-klein-4b",
                            torch_dtype=torch.bfloat16,
                            local_files_only=True
                        )
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        if device == "cuda":
                            pipe.enable_sequential_cpu_offload()
                            torch.cuda.empty_cache()
                        else:
                            st.warning("CUDA non détecté : exécution sur CPU (beaucoup plus lent).")
                            pipe.to("cpu")
                        return pipe, device

                    pipe, device = load_flux_pipeline()

                    # Seed aléatoire pour chaque génération
                    seed = random.randint(0, 999999999)
                    generator = torch.Generator(device=device).manual_seed(seed)

                    # Génération de l'image (paramètres simples et rapides)
                    image = pipe(
                        prompt=user_prompt,
                        height=768,
                        width=1024,
                        guidance_scale=1.0,
                        num_inference_steps=4,
                        generator=generator
                    ).images[0]

                    # Affichage de l'image
                    st.image(image, caption=user_prompt, use_container_width=True)

                    # Bouton de téléchargement
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


if __name__ == "__main__":
    # Pour tester le plugin indépendamment (optionnel)
    st.write("Flux Plugin standalone test")
