from lib.global_vars import translations, t
from app import Widget
import streamlit as st
from typing import List, Dict, Any, Optional
from widgets.product_selector import ProductSelectorWidget
import random
import re
from collections import Counter

# Update translations
translations["en"].update({
    "subjectselector_header": "Subject Selector",
    "subjectselector_select_product": "Select a Product",
    "subjectselector_random_product": "Random Product",
    "subjectselector_product_stats": "Product: {title} ({chars} chars, {words} words, {chapters} chapters, {subchapters} subchapters)",
    "subjectselector_no_chapters": "No chapters detected in the product content.",
    "subjectselector_select_chapter": "Select a Chapter",
    "subjectselector_random_chapter": "Random Chapter",
    "subjectselector_chapter_stats": "Chapter: {title} ({chars} chars, {words} words, {subchapters} subchapters)",
    "subjectselector_no_subchapters": "No subchapters detected in the chapter.",
    "subjectselector_select_subchapter": "Select a Subchapter",
    "subjectselector_random_subchapter": "Random Subchapter",
    "subjectselector_subchapter_stats": "Subchapter: {title} ({chars} chars, {words} words)",
    "subjectselector_use_whole": "Use Entire Selection",
    "subjectselector_extract_subject": "Extract Random Subject",
    "subjectselector_extracting": "Extracting subject...",
    "subjectselector_no_content": "No content available for subject extraction."
})

translations["fr"].update({
    "subjectselector_header": "Sélecteur de Sujet",
    "subjectselector_select_product": "Sélectionner un Produit",
    "subjectselector_random_product": "Produit Aléatoire",
    "subjectselector_product_stats": "Produit : {title} ({chars} caractères, {words} mots, {chapters} chapitres, {subchapters} sous-chapitres)",
    "subjectselector_no_chapters": "Aucun chapitre détecté dans le contenu du produit.",
    "subjectselector_select_chapter": "Sélectionner un Chapitre",
    "subjectselector_random_chapter": "Chapitre Aléatoire",
    "subjectselector_chapter_stats": "Chapitre : {title} ({chars} caractères, {words} mots, {subchapters} sous-chapitres)",
    "subjectselector_no_subchapters": "Aucun sous-chapitre détecté dans le chapitre.",
    "subjectselector_select_subchapter": "Sélectionner un Sous-chapitre",
    "subjectselector_random_subchapter": "Sous-chapitre Aléatoire",
    "subjectselector_subchapter_stats": "Sous-chapitre : {title} ({chars} caractères, {words} mots)",
    "subjectselector_use_whole": "Utiliser la Sélection Entière",
    "subjectselector_extract_subject": "Extraire un Sujet Aléatoire",
    "subjectselector_extracting": "Extraction du sujet...",
    "subjectselector_no_content": "Aucun contenu disponible pour l'extraction de sujet."
})

class SubjectSelectorWidget(Widget):
    def __init__(self, name: str, prefix: str, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self.product_selector = ProductSelectorWidget("product_selector", f"{self.prefix}_product", plugin_manager)
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize session state for subject selection."""
        state_keys = [
            f'{self.prefix}_selected_product',
            f'{self.prefix}_selected_chapter',
            f'{self.prefix}_selected_subchapter',
            f'{self.prefix}_selected_content'
        ]
        for key in state_keys:
            if key not in st.session_state:
                st.session_state[key] = None

    def _extract_chapters(self, content: str) -> List[Dict[str, Any]]:
        """Extract chapters from content based on markdown headers (##)."""
        chapters = []
        # Split content by ## headers
        sections = re.split(r'(^##\s+.*$)', content, flags=re.MULTILINE)
        for i in range(1, len(sections), 2):  # Start from 1 to get headers
            title = sections[i].replace('##', '').strip()
            chapter_content = sections[i + 1].strip() if i + 1 < len(sections) else ""
            chapters.append({
                'title': title,
                'content': chapter_content,
                'keywords': self._extract_keywords(chapter_content)
            })
        return chapters

    def _extract_subchapters(self, content: str) -> List[Dict[str, Any]]:
        """Extract subchapters from chapter content based on markdown headers (###)."""
        subchapters = []
        sections = re.split(r'(^###\s+.*$)', content, flags=re.MULTILINE)
        for i in range(1, len(sections), 2):
            title = sections[i].replace('###', '').strip()
            subchapter_content = sections[i + 1].strip() if i + 1 < len(sections) else ""
            subchapters.append({
                'title': title,
                'content': subchapter_content,
                'keywords': self._extract_keywords(subchapter_content)
            })
        return subchapters

    def _extract_keywords(self, content: str) -> str:
        """Extract keywords from content (simple word frequency-based approach)."""
        words = re.findall(r'\b\w+\b', content.lower())
        word_counts = Counter(words)
        # Filter out common stop words (simplified list)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word, count in word_counts.most_common(10) if word not in stop_words and len(word) > 3]
        return ', '.join(keywords)

    def _get_content_stats(self, content: str) -> Dict[str, int]:
        """Calculate statistics for content."""
        chars = len(content)
        words = len(re.findall(r'\b\w+\b', content))
        chapters = len(self._extract_chapters(content))
        subchapters = sum(len(self._extract_subchapters(ch['content'])) for ch in self._extract_chapters(content))
        return {'chars': chars, 'words': words, 'chapters': chapters, 'subchapters': subchapters}

    def _extract_random_subject(self, content: str, title: str) -> Dict[str, Any]:
        """Use LLM to extract a random subject from content."""
        if not content:
            st.error(t("subjectselector_no_content"))
            return {'title': title, 'content': '', 'keywords': ''}

        prompt = (
            "You are an expert in content analysis. From the provided text, identify and extract a concise subject or topic. "
            "Return a title (max 50 chars) and a summary (max 200 chars) of the subject. "
            "Focus on a single, specific idea or theme present in the text. "
            "Format the response as: Title: <title>\nSummary: <summary>\nKeywords: <keywords>"
        )
        response = self.process_with_llm(
            f"Title: {title}\nContent: {content[:2000]}",  # Limit content to avoid LLM token limits
            sysprompt=prompt
        )
        # Parse response
        lines = response.split('\n')
        subject_title = title
        subject_content = ""
        keywords = ""
        for line in lines:
            if line.startswith('Title:'):
                subject_title = line.replace('Title:', '').strip()[:50]
            elif line.startswith('Summary:'):
                subject_content = line.replace('Summary:', '').strip()[:200]
            elif line.startswith('Keywords:'):
                keywords = line.replace('Keywords:', '').strip()

        return {
            'title': subject_title,
            'content': subject_content,
            'keywords': keywords
        }

    def display(self) -> Optional[Dict[str, Any]]:
        """Display the subject selector and return the selected subject."""
        st.header(t("subjectselector_header"))

        # Step 1: Product selection
        selected_product = self.product_selector.display()
        if not selected_product:
            return None

        # Initialize session state for product if changed
        if st.session_state.get(f'{self.prefix}_selected_product') != selected_product:
            st.session_state[f'{self.prefix}_selected_product'] = selected_product
            st.session_state[f'{self.prefix}_selected_chapter'] = None
            st.session_state[f'{self.prefix}_selected_subchapter'] = None
            st.session_state[f'{self.prefix}_selected_content'] = None

        # Step 2: Display product stats and excerpt
        product_stats = self._get_content_stats(selected_product['content'])
        st.subheader(t("subjectselector_product_stats").format(
            title=selected_product['title'],
            chars=product_stats['chars'],
            words=product_stats['words'],
            chapters=product_stats['chapters'],
            subchapters=product_stats['subchapters']
        ))
        with st.expander("Product Excerpt"):
            st.markdown(selected_product['content'][:1000] + ('...' if len(selected_product['content']) > 1000 else ''))

        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(t("subjectselector_use_whole"), key=f"{self.prefix}_use_product"):
                st.session_state[f'{self.prefix}_selected_content'] = selected_product
                return selected_product
        with col2:
            if st.button(t("subjectselector_extract_subject"), key=f"{self.prefix}_extract_product_subject"):
                with st.spinner(t("subjectselector_extracting")):
                    subject = self._extract_random_subject(selected_product['content'], selected_product['title'])
                    st.session_state[f'{self.prefix}_selected_content'] = subject
                    return subject

        # Step 3: Chapter selection (if chapters exist)
        chapters = self._extract_chapters(selected_product['content'])
        if not chapters:
            st.info(t("subjectselector_no_chapters"))
            return selected_product if st.session_state.get(f'{self.prefix}_selected_content') == selected_product else None

        chapter_titles = [ch['title'] for ch in chapters]
        default_chapter = st.session_state.get(f'{self.prefix}_selected_chapter')
        default_index = chapter_titles.index(default_chapter['title']) if default_chapter and default_chapter['title'] in chapter_titles else 0

        col1, col2 = st.columns([3, 1])
        with col1:
            selected_chapter_title = st.selectbox(
                t("subjectselector_select_chapter"),
                options=chapter_titles,
                index=default_index,
                key=f"{self.prefix}_chapter_select"
            )
        with col2:
            if st.button(t("subjectselector_random_chapter"), key=f"{self.prefix}_random_chapter"):
                selected_chapter = random.choice(chapters)
                st.session_state[f'{self.prefix}_selected_chapter'] = selected_chapter
                st.session_state[f'{self.prefix}_selected_subchapter'] = None
                st.rerun()

        selected_chapter = next((ch for ch in chapters if ch['title'] == selected_chapter_title), chapters[0])
        if selected_chapter != st.session_state.get(f'{self.prefix}_selected_chapter'):
            st.session_state[f'{self.prefix}_selected_chapter'] = selected_chapter
            st.session_state[f'{self.prefix}_selected_subchapter'] = None

        # Display chapter stats and excerpt
        chapter_stats = self._get_content_stats(selected_chapter['content'])
        st.subheader(t("subjectselector_chapter_stats").format(
            title=selected_chapter['title'],
            chars=chapter_stats['chars'],
            words=chapter_stats['words'],
            subchapters=chapter_stats['subchapters']
        ))
        with st.expander("Chapter Excerpt"):
            st.markdown(selected_chapter['content'][:1000] + ('...' if len(selected_chapter['content']) > 1000 else ''))

        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(t("subjectselector_use_whole"), key=f"{self.prefix}_use_chapter"):
                st.session_state[f'{self.prefix}_selected_content'] = selected_chapter
                return selected_chapter
        with col2:
            if st.button(t("subjectselector_extract_subject"), key=f"{self.prefix}_extract_chapter_subject"):
                with st.spinner(t("subjectselector_extracting")):
                    subject = self._extract_random_subject(selected_chapter['content'], selected_chapter['title'])
                    st.session_state[f'{self.prefix}_selected_content'] = subject
                    return subject

        # Step 4: Subchapter selection (if subchapters exist)
        subchapters = self._extract_subchapters(selected_chapter['content'])
        if not subchapters:
            st.info(t("subjectselector_no_subchapters"))
            return selected_chapter if st.session_state.get(f'{self.prefix}_selected_content') == selected_chapter else None

        subchapter_titles = [sc['title'] for sc in subchapters]
        default_subchapter = st.session_state.get(f'{self.prefix}_selected_subchapter')
        default_index = subchapter_titles.index(default_subchapter['title']) if default_subchapter and default_subchapter['title'] in subchapter_titles else 0

        col1, col2 = st.columns([3, 1])
        with col1:
            selected_subchapter_title = st.selectbox(
                t("subjectselector_select_subchapter"),
                options=subchapter_titles,
                index=default_index,
                key=f"{self.prefix}_subchapter_select"
            )
        with col2:
            if st.button(t("subjectselector_random_subchapter"), key=f"{self.prefix}_random_subchapter"):
                selected_subchapter = random.choice(subchapters)
                st.session_state[f'{self.prefix}_selected_subchapter'] = selected_subchapter
                st.rerun()

        selected_subchapter = next((sc for sc in subchapters if sc['title'] == selected_subchapter_title), subchapters[0])
        if selected_subchapter != st.session_state.get(f'{self.prefix}_selected_subchapter'):
            st.session_state[f'{self.prefix}_selected_subchapter'] = selected_subchapter

        # Display subchapter stats and excerpt
        subchapter_stats = self._get_content_stats(selected_subchapter['content'])
        st.subheader(t("subjectselector_subchapter_stats").format(
            title=selected_subchapter['title'],
            chars=subchapter_stats['chars'],
            words=subchapter_stats['words']
        ))
        with st.expander("Subchapter Excerpt"):
            st.markdown(selected_subchapter['content'][:1000] + ('...' if len(selected_subchapter['content']) > 1000 else ''))

        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(t("subjectselector_use_whole"), key=f"{self.prefix}_use_subchapter"):
                st.session_state[f'{self.prefix}_selected_content'] = selected_subchapter
                return selected_subchapter
        with col2:
            if st.button(t("subjectselector_extract_subject"), key=f"{self.prefix}_extract_subchapter_subject"):
                with st.spinner(t("subjectselector_extracting")):
                    subject = self._extract_random_subject(selected_subchapter['content'], selected_subchapter['title'])
                    st.session_state[f'{self.prefix}_selected_content'] = subject
                    return subject

        return st.session_state.get(f'{self.prefix}_selected_content')
