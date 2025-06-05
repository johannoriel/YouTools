# lib/movies_commands.py

from abc import ABC, abstractmethod
from moviepy import VideoFileClip, ColorClip, AudioClip, concatenate_videoclips
import numpy as np
import os
from typing import Tuple, Dict, Optional

from moviepy import VideoFileClip, AudioClip
from lib.video_utils import *
from lib.video_anim import replace_with_image
import re

def debug_video_clip(clip, level=0, title="Video Clip"):
    """
    Affiche une hiérarchie détaillée d'un clip vidéo dans Streamlit sous forme de Markdown,
    incluant les sous-clips, leurs dimensions, et les informations sur l'audio.

    Args:
        clip: Le clip vidéo (VideoFileClip, CompositeVideoClip, etc.) à diagnostiquer.
        level: Niveau d'indentation pour la hiérarchie (utilisé récursivement).
        title: Titre à afficher pour ce clip dans la description.

    Returns:
        str: Chaîne Markdown représentant la hiérarchie du clip.
    """
    indent = "  " * level
    markdown = []

    # Titre du clip
    markdown.append(f"{indent}- **{title} (Type: {type(clip).__name__})**")

    # Informations générales
    markdown.append(f"{indent}  - **Type**: {type(clip).__name__}")
    markdown.append(
        f"{indent}  - **Duration**: {clip.duration:.2f} seconds" if clip.duration
        else f"{indent}  - **Duration**: None"
    )
    markdown.append(
        f"{indent}  - **Dimensions**: {clip.w}x{clip.h}" if hasattr(clip, 'w') and hasattr(clip, 'h')
        else f"{indent}  - **Dimensions**: Unknown"
    )

    # Informations sur l'audio
    if hasattr(clip, 'audio') and clip.audio is not None:
        markdown.append(f"{indent}  - **Audio**: Present")
        markdown.append(
            f"{indent}  - **Audio Duration**: {clip.audio.duration:.2f} seconds" if clip.audio.duration
            else f"{indent}  - **Audio Duration**: None"
        )
        markdown.append(
            f"{indent}  - **Audio FPS**: {clip.audio.fps}" if hasattr(clip.audio, 'fps')
            else f"{indent}  - **Audio FPS**: Unknown"
        )
    else:
        markdown.append(f"{indent}  - **Audio**: None")

    # Gestion des sous-clips
    sub_clips = []
    if isinstance(clip, CompositeVideoClip) and hasattr(clip, 'clips'):
        sub_clips = clip.clips
        markdown.append(f"{indent}  - **Sub-clips**: {len(sub_clips)} (Composite)")
    elif hasattr(clip, 'clip') and clip.clip is not None:
        sub_clips = [clip.clip]
        markdown.append(f"{indent}  - **Sub-clip**: 1 (Single wrapped clip)")
    else:
        markdown.append(f"{indent}  - **Sub-clips**: None")

    # Ajouter les sous-clips récursivement
    for i, sub_clip in enumerate(sub_clips):
        sub_markdown = debug_video_clip(sub_clip, level + 1, f"Sub-clip {i + 1}")
        markdown.append(sub_markdown)

    # Retourner la chaîne Markdown pour ce niveau
    return "\n".join(markdown)

def display_video_clip_debug(clip, title="Video Clip"):
    """
    Affiche le débogage d'un clip vidéo dans Streamlit en utilisant Markdown.

    Args:
        clip: Le clip vidéo à diagnostiquer.
        title: Titre principal pour l'affichage.
    """
    markdown = debug_video_clip(clip, level=0, title=title)
    st.markdown(markdown)

class MovieCommand(ABC):
    """Classe de base pour les commandes de montage vidéo."""

    @abstractmethod
    def get_label(self) -> str:
        """Retourne le libellé de la commande pour le selectbox."""
        pass

    @abstractmethod
    def is_enabled(self, has_selection: bool, has_text: bool, media_type: Optional[str]) -> bool:
        """Vérifie si la commande peut être activée selon les paramètres disponibles.

        Args:
            has_selection: Indique si une sélection temporelle est présente
            has_text: Indique si un texte est saisi
            media_type: Type de média sélectionné ('image', 'video', 'audio', None)
        """
        pass

    @abstractmethod
    def get_default_command(self, start_time: str, end_time: Optional[str],
                          text: Optional[str], media_path: Optional[str]) -> str:
        """Génère la commande par défaut avec les paramètres fournis.

        Args:
            start_time: Timecode de début
            end_time: Timecode de fin (optionnel pour certaines commandes)
            text: Texte saisi (optionnel)
            media_path: Chemin du média sélectionné (optionnel)
        """
        pass

    @abstractmethod
    def execute(self, clip: VideoFileClip, command_line: str,
                target_size: Tuple[int, int], **kwargs) -> Tuple[VideoFileClip, float]:
        """Exécute la commande sur le clip vidéo.

        Args:
            clip: Clip vidéo d'entrée
            command_line: Ligne de commande complète
            target_size: Taille cible du clip (largeur, hauteur)
            **kwargs: Arguments supplémentaires (ex: font, font_size, background_type)

        Returns:
            Tuple contenant le clip modifié et le delta de durée (en secondes)
        """
        pass

    def _parse_timecode(self, timecode: str) -> float:
        """Parse un timecode HH:MM:SS.mmm en secondes."""
        try:
            h, m, s = map(float, timecode.replace(",", ".").split(":"))
            return h * 3600 + m * 60 + s
        except ValueError:
            raise ValueError(f"Invalid timecode format: {timecode}")



class CommandOrchestrator:
    """Orchestre l'exécution des commandes de montage vidéo."""

    def _is_valid_timecode(self, timecode: str) -> bool:
        try:
            # Remplacer la virgule par un point si nécessaire
            normalized = timecode.replace(",", ".")
            hh, mm, ss = normalized.split(":")

            # Vérifier la longueur de chaque partie
            if len(hh) != 2 or len(mm) != 2 or len(ss) != 6:  # "SS.mmm" = 6 caractères
                return False

            # Convertir en nombres et vérifier les plages
            h = int(hh)
            m = int(mm)
            s = float(ss)

            return (0 <= h <= 23 and 0 <= m <= 59 and 0 <= s < 60.0)
        except (ValueError, AttributeError):
            return False

    def __init__(self):
        self.commands: Dict[str, MovieCommand] = {}
        self.register_command("CHANGE_VIDEO", ChangeVideoCommand())
        self.register_command("replace_image", ReplaceImageCommand())
        self.register_command("insert_video", InsertVideoCommand())
        self.register_command("insertVideoWithText", InsertVideoWithTextCommand())
        self.register_command("replace_video", ReplaceVideoCommand())
        self.register_command("replace_video_keep_audio", ReplaceVideoKeepAudioCommand())
        self.register_command("addtext", AddTextCommand())
        self.register_command("addBottomText", AddBottomTextCommand())
        self.register_command("remove_section", RemoveSectionCommand())
        self.register_command("replace_audio", ReplaceAudioCommand())
        self.register_command("insert_audio", InsertAudioCommand())

    def register_command(self, command_name: str, command: MovieCommand):
        """Enregistre une nouvelle commande."""
        self.commands[command_name] = command

    def get_available_commands(self, has_selection: bool, has_text: bool,
                            media_type: Optional[str]) -> Dict[str, MovieCommand]:
        """Retourne les commandes disponibles selon le contexte."""
        return {
            name: cmd for name, cmd in self.commands.items()
            if cmd.is_enabled(has_selection, has_text, media_type)
        }

    def execute_operations(self, video_path: str, operations: str,
                         target_size: Tuple[int, int], **kwargs) -> Tuple[VideoFileClip, list]:
        """Exécute une série d'opérations sur un clip vidéo.

        Args:
            video_path: Chemin du fichier vidéo initial
            operations: Chaîne contenant les opérations (une par ligne)
            target_size: Taille cible du clip
            **kwargs: Arguments supplémentaires pour les commandes

        Returns:
            Tuple contenant le clip final et le journal des opérations
        """
        clips = []
        current_clip = None
        duration_offset = 0
        global_time_offset = 0
        operation_log = []

        # Nettoyer et préparer les opérations
        ops_list = [op.split("//")[0].strip() for op in operations.split("\n") if op.strip()]

        # Vérifier si la première commande est CHANGE_VIDEO
        if not ops_list or not ops_list[0].startswith("CHANGE_VIDEO"):
            # Insérer une commande CHANGE_VIDEO artificielle avec video_path
            video_name = os.path.basename(video_path)
            ops_list.insert(0, f"CHANGE_VIDEO {video_name}")

        for op_cleaned in ops_list:
            parts = op_cleaned.split(maxsplit=1)
            cmd = parts[0]

            if cmd in self.commands:
                command = self.commands[cmd]
                try:
                    # Si c'est CHANGE_VIDEO, le clip actuel est remplacé
                    if cmd == "CHANGE_VIDEO":
                        if current_clip is not None:
                            clips.append(current_clip)
                            global_time_offset += current_clip.duration if current_clip else 0
                        duration_offset = 0
                    elif current_clip is None:
                        raise ValueError("No video loaded. A CHANGE_VIDEO command must be executed first.")

                    # Exécuter la commande
                    current_clip, duration_change = command.execute(
                        current_clip, op_cleaned, target_size, **kwargs)

                    # Calculer les timecodes réels
                    start_time = parts[1].split()[0] if len(parts) > 1 else ""
                    start_sec = 0
                    if cmd != "CHANGE_VIDEO" and start_time:
                        start_sec = self._parse_timecode(start_time) + duration_offset
                    real_start = self._format_timecode(start_sec + global_time_offset)

                    # Extraire end_time si pertinent
                    end_sec = start_sec
                    if len(parts) > 1 and len(parts[1].split()) > 1 and cmd != "CHANGE_VIDEO":
                        end_time = parts[1].split()[1]
                        if self._is_valid_timecode(end_time):
                            end_sec = self._parse_timecode(end_time) + duration_offset
                    real_end = self._format_timecode(end_sec + global_time_offset) if end_sec else None
                    duration_str = f"{(end_sec - start_sec):.3f}s" if end_sec else ""
                    duration_offset += duration_change

                    operation_log.append({
                        "Nature": cmd,
                        "Details": " ".join(parts[1:]) if len(parts) > 1 else "",
                        "Start": real_start,
                        "End": real_end,
                        "Duration": duration_str
                    })
                except Exception as e:
                    raise e
                    raise ValueError(f"Error executing command {cmd}: {str(e)}")
            else:
                raise ValueError(f"Unknown command: {cmd}")

        if current_clip is None:
            raise ValueError("No video was loaded during execution.")

        clips.append(current_clip)
        final_clip = concatenate_videoclips(clips, method="compose")

        for clip in clips:
            clip.close()

        return final_clip, operation_log

    def _parse_timecode(self, timecode: str) -> float:
        """Parse un timecode HH:MM:SS.mmm en secondes."""
        try:
            # Remplacer les virgules par des points pour les décimales
            timecode = timecode.replace(",", ".")

            # Séparer les composants
            parts = timecode.split(":")
            if len(parts) != 3:
                raise ValueError("Le timecode doit avoir le format HH:MM:SS.mmm :"+timecode)

            h = float(parts[0])
            m = float(parts[1])
            s = float(parts[2])

            return h * 3600 + m * 60 + s
        except ValueError as e:
            raise ValueError(f"Invalid timecode format: {timecode} - {str(e)}")

    def _format_timecode(self, seconds: float) -> str:
        """Formate les secondes en timecode HH:MM:SS.mmm."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


# Classes pour chaque commande
class ReplaceImageCommand(MovieCommand):
    def get_label(self) -> str:
        return "Replace with Image"

    def is_enabled(self, has_selection: bool, has_text: bool, media_type: Optional[str]) -> bool:
        return has_selection and media_type == "image"

    def get_default_command(self, start_time: str, end_time: Optional[str],
                          text: Optional[str], media_path: Optional[str]) -> str:
        if not end_time or not media_path:
            raise ValueError("End time and media path required for replace_image")
        return f"replace_image {start_time} {end_time} {media_path}"

    def execute(self, clip: VideoFileClip, command_line: str,
                target_size: Tuple[int, int], **kwargs) -> Tuple[VideoFileClip, float]:
        parts = command_line.split(maxsplit=3)
        if len(parts) < 4:
            raise ValueError(f"Invalid replace_image command: {command_line}")

        start_time, end_time, image_path = parts[1], parts[2], parts[3]
        start_sec = self._parse_timecode(start_time)
        end_sec = self._parse_timecode(end_time)
        background_type = "green" if kwargs.get("use_green_background", False) else "video"

        modified_clip = replace_with_image(
            clip, start_sec, end_sec, image_path, target_size, background=background_type)
        return modified_clip, 0

class InsertVideoCommand(MovieCommand):
    def get_label(self) -> str:
        return "Insert Video"

    def is_enabled(self, has_selection: bool, has_text: bool, media_type: Optional[str]) -> bool:
        return has_selection and media_type == "video"

    def get_default_command(self, start_time: str, end_time: Optional[str],
                          text: Optional[str], media_path: Optional[str]) -> str:
        if not media_path:
            raise ValueError("Media path required for insert_video")
        return f"insert_video {start_time} {media_path}"

    def execute(self, clip: VideoFileClip, command_line: str,
                target_size: Tuple[int, int], **kwargs) -> Tuple[VideoFileClip, float]:
        parts = command_line.split(maxsplit=2)
        if len(parts) < 2:
            raise ValueError(f"Invalid insert_video command: {command_line}")

        start_time = parts[1]
        # Vérifier si start_time est un timecode valide
        try:
            start_sec = self._parse_timecode(start_time)
        except ValueError:
            raise ValueError(f"Invalid timecode format: {start_time}")

        if len(parts) < 3:
            raise ValueError(f"No video path provided in insert_video command: {command_line}")
        video_path = parts[2]

        modified_clip, duration_change = insert_video(
            clip, start_sec, video_path, target_size)
        return modified_clip, duration_change

class InsertVideoWithTextCommand(MovieCommand):
    def get_label(self) -> str:
        return "Insert Video with Text"

    def is_enabled(self, has_selection: bool, has_text: bool, media_type: Optional[str]) -> bool:
        return has_selection and has_text and media_type == "video"

    def get_default_command(self, start_time: str, end_time: Optional[str],
                          text: Optional[str], media_path: Optional[str]) -> str:
        if not media_path or not text:
            raise ValueError("Media path and text required for insertVideoWithText")
        text_command = text.replace("\n", "\\")
        return f"insertVideoWithText {start_time} {media_path} | {text_command}"

    def execute(self, clip: VideoFileClip, command_line: str,
                target_size: Tuple[int, int], **kwargs) -> Tuple[VideoFileClip, float]:
        match = re.match(r"insertVideoWithText\s+(\S+)\s+(.+?)\s+\|\s+(.+)", command_line)
        if not match:
            raise ValueError(f"Invalid insertVideoWithText command: {command_line}")

        start_time, video_path, text = match.groups()
        start_sec = self._parse_timecode(start_time)

        modified_clip, duration_change = insert_video_with_text(
            clip, start_sec, video_path, text, target_size,
            font=kwargs.get("font"), font_size=kwargs.get("font_size"),
            use_green_background=kwargs.get("use_green_background", False),
            text_style=kwargs.get("text_style", "outline"))
        return modified_clip, duration_change

class ReplaceVideoCommand(MovieCommand):
    def get_label(self) -> str:
        return "Replace with Video"

    def is_enabled(self, has_selection: bool, has_text: bool, media_type: Optional[str]) -> bool:
        return has_selection and media_type == "video"

    def get_default_command(self, start_time: str, end_time: Optional[str],
                          text: Optional[str], media_path: Optional[str]) -> str:
        if not end_time or not media_path:
            raise ValueError("End time and media path required for replace_video")
        return f"replace_video {start_time} {end_time} {media_path}"

    def execute(self, clip: VideoFileClip, command_line: str,
                target_size: Tuple[int, int], **kwargs) -> Tuple[VideoFileClip, float]:
        parts = command_line.split(maxsplit=3)
        if len(parts) < 4:
            raise ValueError(f"Invalid replace_video command: {command_line}")

        start_time, end_time, video_path = parts[1], parts[2], parts[3]
        start_sec = self._parse_timecode(start_time)
        end_sec = self._parse_timecode(end_time)
        background_type = "green" if kwargs.get("use_green_background", False) else "video"

        modified_clip, duration_change = replace_with_video(
            clip, start_sec, end_sec, video_path, target_size, background=background_type)
        return modified_clip, duration_change

class ReplaceVideoKeepAudioCommand(MovieCommand):
    def get_label(self) -> str:
        return "Replace Video (Keep Audio)"

    def is_enabled(self, has_selection: bool, has_text: bool, media_type: Optional[str]) -> bool:
        return has_selection and media_type == "video"

    def get_default_command(self, start_time: str, end_time: Optional[str],
                          text: Optional[str], media_path: Optional[str]) -> str:
        if not end_time or not media_path:
            raise ValueError("End time and media path required for replace_video_keep_audio")
        return f"replace_video_keep_audio {start_time} {end_time} {media_path}"

    def execute(self, clip: VideoFileClip, command_line: str,
                target_size: Tuple[int, int], **kwargs) -> Tuple[VideoFileClip, float]:
        parts = command_line.split(maxsplit=3)
        if len(parts) < 4:
            raise ValueError(f"Invalid replace_video_keep_audio command: {command_line}")

        start_time, end_time, video_path = parts[1], parts[2], parts[3]
        start_sec = self._parse_timecode(start_time)
        end_sec = self._parse_timecode(end_time)
        background_type = "green" if kwargs.get("use_green_background", False) else "video"

        modified_clip = replace_video_keep_audio(
            clip, start_sec, end_sec, video_path, target_size, background=background_type)
        return modified_clip, 0

class AddTextCommand(MovieCommand):
    def get_label(self) -> str:
        return "Add Animated Text"

    def is_enabled(self, has_selection: bool, has_text: bool, media_type: Optional[str]) -> bool:
        return has_selection and has_text

    def get_default_command(self, start_time: str, end_time: Optional[str],
                          text: Optional[str], media_path: Optional[str]) -> str:
        if not end_time or not text:
            raise ValueError("End time and text required for addtext")
        text_command = text.replace("\n", "\\")
        return f"addtext {start_time} {end_time} fromLeft 1s {text_command}"

    def execute(self, clip: VideoFileClip, command_line: str,
                target_size: Tuple[int, int], **kwargs) -> Tuple[VideoFileClip, float]:
        parts = command_line.split(maxsplit=5)
        if len(parts) < 6:
            raise ValueError(f"Invalid addtext command: {command_line}")

        start_time, end_time, animation_type, anim_duration, text = parts[1], parts[2], parts[3], parts[4], parts[5]
        start_sec = self._parse_timecode(start_time)
        end_sec = self._parse_timecode(end_time)
        anim_duration_sec = float(anim_duration[:-1])

        modified_clip = add_animated_text(
            clip, start_sec, end_sec, text, animation_type, anim_duration_sec, target_size,
            font=kwargs.get("font"), font_size=kwargs.get("font_size"),
            use_green_background=kwargs.get("use_green_background", False),
            position="center", text_style=kwargs.get("text_style", "outline"))
        return modified_clip, 0

class AddBottomTextCommand(MovieCommand):
    def get_label(self) -> str:
        return "Add Bottom Text"

    def is_enabled(self, has_selection: bool, has_text: bool, media_type: Optional[str]) -> bool:
        return has_selection and has_text

    def get_default_command(self, start_time: str, end_time: Optional[str],
                          text: Optional[str], media_path: Optional[str]) -> str:
        if not end_time or not text:
            raise ValueError("End time and text required for addBottomText")
        text_command = text.replace("\n", "\\")
        return f"addBottomText {start_time} {end_time} fromLeft 1s {text_command}"

    def execute(self, clip: VideoFileClip, command_line: str,
                target_size: Tuple[int, int], **kwargs) -> Tuple[VideoFileClip, float]:
        parts = command_line.split(maxsplit=5)
        if len(parts) < 6:
            raise ValueError(f"Invalid addBottomText command: {command_line}")

        start_time, end_time, animation_type, anim_duration, text = parts[1], parts[2], parts[3], parts[4], parts[5]
        start_sec = self._parse_timecode(start_time)
        end_sec = self._parse_timecode(end_time)
        anim_duration_sec = float(anim_duration[:-1])

        modified_clip = add_animated_text(
            clip, start_sec, end_sec, text, animation_type, anim_duration_sec, target_size,
            font=kwargs.get("font"), font_size=kwargs.get("font_size"),
            use_green_background=kwargs.get("use_green_background", False),
            position="bottom", text_style=kwargs.get("text_style", "outline"))
        return modified_clip, 0

class RemoveSectionCommand(MovieCommand):
    def get_label(self) -> str:
        return "Remove Section"

    def is_enabled(self, has_selection: bool, has_text: bool, media_type: Optional[str]) -> bool:
        return has_selection

    def get_default_command(self, start_time: str, end_time: Optional[str],
                          text: Optional[str], media_path: Optional[str]) -> str:
        if not end_time:
            raise ValueError("End time required for remove_section")
        return f"remove_section {start_time} {end_time}"

    def execute(self, clip: VideoFileClip, command_line: str,
                target_size: Tuple[int, int], **kwargs) -> Tuple[VideoFileClip, float]:
        parts = command_line.split(maxsplit=2)
        if len(parts) < 3:
            raise ValueError(f"Invalid remove_section command: {command_line}")

        start_time, end_time = parts[1], parts[2]
        start_sec = self._parse_timecode(start_time)
        end_sec = self._parse_timecode(end_time)

        modified_clip, duration_change = remove_section(clip, start_sec, end_sec)
        if modified_clip.audio is None:
            modified_clip = modified_clip.with_audio(
                AudioClip(lambda t: np.zeros((int(t * 44100), 2)), duration=modified_clip.duration))
        return modified_clip, duration_change

class ReplaceAudioCommand(MovieCommand):
    def get_label(self) -> str:
        return "Replace Audio"

    def is_enabled(self, has_selection: bool, has_text: bool, media_type: Optional[str]) -> bool:
        return has_selection and media_type == "audio"

    def get_default_command(self, start_time: str, end_time: Optional[str],
                          text: Optional[str], media_path: Optional[str]) -> str:
        if not end_time or not media_path:
            raise ValueError("End time and media path required for replace_audio")
        return f"replace_audio {start_time} {end_time} {media_path}"

    def execute(self, clip: VideoFileClip, command_line: str,
                target_size: Tuple[int, int], **kwargs) -> Tuple[VideoFileClip, float]:
        parts = command_line.split(maxsplit=3)
        if len(parts) < 4:
            raise ValueError(f"Invalid replace_audio command: {command_line}")

        start_time, end_time, audio_path = parts[1], parts[2], parts[3]
        start_sec = self._parse_timecode(start_time)
        end_sec = self._parse_timecode(end_time)

        modified_clip, duration_change = replace_audio(
            clip, start_sec, end_sec, audio_path, target_size)
        return modified_clip, duration_change

class InsertAudioCommand(MovieCommand):
    def get_label(self) -> str:
        return "Insert Audio"

    def is_enabled(self, has_selection: bool, has_text: bool, media_type: Optional[str]) -> bool:
        return has_selection and media_type == "audio"

    def get_default_command(self, start_time: str, end_time: Optional[str],
                          text: Optional[str], media_path: Optional[str]) -> str:
        if not media_path:
            raise ValueError("Media path required for insert_audio")
        return f"insert_audio {start_time} {media_path}"

    def execute(self, clip: VideoFileClip, command_line: str,
                target_size: Tuple[int, int], **kwargs) -> Tuple[VideoFileClip, float]:
        parts = command_line.split(maxsplit=2)
        if len(parts) < 3:
            raise ValueError(f"Invalid insert_audio command: {command_line}")

        start_time, audio_path = parts[1], parts[2]
        start_sec = self._parse_timecode(start_time)

        modified_clip, duration_change = insert_audio(
            clip, start_sec, audio_path, target_size)
        return modified_clip, duration_change

class ChangeVideoCommand(MovieCommand):
    def get_label(self) -> str:
        return "Change Video"

    def is_enabled(self, has_selection: bool, has_text: bool, media_type: Optional[str]) -> bool:
        return media_type == "video"

    def get_default_command(self, start_time: str, end_time: Optional[str],
                          text: Optional[str], media_path: Optional[str]) -> str:
        if not media_path:
            raise ValueError("Media path required for CHANGE_VIDEO")
        video_name = os.path.basename(media_path)
        return f"CHANGE_VIDEO {video_name}"

    def execute(self, clip: VideoFileClip, command_line: str,
                target_size: Tuple[int, int], **kwargs) -> Tuple[VideoFileClip, float]:
        parts = command_line.split(maxsplit=1)
        if len(parts) < 2:
            raise ValueError(f"Invalid CHANGE_VIDEO command: {command_line}")

        video_name = parts[1]
        new_video_path = os.path.join(kwargs.get("working_dir", ""), video_name)
        if not os.path.exists(new_video_path):
            raise ValueError(f"Video file not found: {new_video_path}")

        # Charger le nouveau clip
        new_clip = VideoFileClip(new_video_path)
        new_clip = new_clip.resized(target_size)
        new_clip = concatenate_videoclips([new_clip.subclipped(0, new_clip.duration)]) # BUG correction

        # Le clip retourné devient le nouveau clip, avec une durée de 0 pour l'offset
        return new_clip, 0
