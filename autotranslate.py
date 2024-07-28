import argparse
import os
import shutil
import sys

def move_and_rename_file(src, dst):
    shutil.move(src, dst)

def main():
    parser = argparse.ArgumentParser(description="Auto doublage")
    parser.add_argument("input", type=str, help="Input file")
    parser.add_argument("output", type=str, help="Output file")
    parser.add_argument("soni_translate_dir", type=str, help="SoniTranslate directory")

    # Ajouter des arguments optionnels
    parser.add_argument("-l", "--lang", type=str, default="French (fr)", help="Target language")
    parser.add_argument("-s", "--speakers", type=int, help="Speakers numbers - unused for the moment")

    args = parser.parse_args()
    os.chdir(args.soni_translate_dir)
    # Ajouter le répertoire SoniTranslate au chemin de recherche des modules
    sys.path.append(args.soni_translate_dir)

    from app_rvc import SoniTranslate, upload_model_list
    from soni_translate.mdx_net import (
        UVR_MODELS,
        MDX_DOWNLOAD_LINK,
        mdxnet_models_dir,
    )
    from soni_translate.utils import (
        remove_files,
        download_list,
        upload_model_list,
        download_manager,
        run_command,
        is_audio_file,
        is_subtitle_file,
        copy_files,
        get_valid_files,
        get_link_list,
        remove_directory_contents,
    )

    for id_model in UVR_MODELS:
        download_manager(
            os.path.join(MDX_DOWNLOAD_LINK, id_model), mdxnet_models_dir
        )

    models_path, index_path = upload_model_list()

    SoniTr = SoniTranslate()

    results = SoniTr.multilingual_media_conversion(
        media_file=args.input,
        target_language=args.lang,
        min_speakers=1,
        max_speakers=3,
        video_output_name=args.output,
        voice_imitation=True,
        burn_subtitles_to_video=True,
    )
    print(results)
    move_and_rename_file(results[0], args.output)

if __name__ == "__main__":
    main()
