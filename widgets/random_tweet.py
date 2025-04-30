from lib.global_vars import translations, t
from app import Widget
import streamlit as st
from typing import List, Dict, Any
from lib.social_api import TwitterAPI
from widgets.product_selector import ProductSelectorWidget

# Ajout des traductions spécifiques au widget
translations["en"].update({
    "randomtweet_tab": "Random Tweet",
    "randomtweet_header": "Generate Random Tweet Thread",
    "randomtweet_generate": "Generate Tweet Thread",
    "randomtweet_generating": "Generating tweet thread...",
    "randomtweet_preview": "Preview Tweet Thread",
    "randomtweet_post": "Post Tweet Thread",
    "randomtweet_posting": "Posting tweet thread...",
    "randomtweet_success": "Tweet thread posted successfully!",
    "randomtweet_error": "Error posting tweet thread: ",
    "randomtweet_add_url": "Add URL Tweet at the End",
    "randomtweet_prompt": "LLM Prompt for Random Tweet Thread",
})

translations["fr"].update({
    "randomtweet_tab": "Tweet Aléatoire",
    "randomtweet_header": "Générer un Fil de Tweets Aléatoire",
    "randomtweet_generate": "Générer le fil de tweets",
    "randomtweet_generating": "Génération du fil de tweets...",
    "randomtweet_preview": "Prévisualiser le fil de tweets",
    "randomtweet_post": "Poster le fil de tweets",
    "randomtweet_posting": "Publication du fil de tweets...",
    "randomtweet_success": "Fil de tweets publié avec succès !",
    "randomtweet_error": "Erreur lors de la publication du fil : ",
    "randomtweet_add_url": "Ajouter un tweet avec l'URL à la fin",
    "randomtweet_prompt": "Prompt LLM pour le fil de tweets aléatoire",
})

class RandomTweetWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self._initialize_session_state()

    def _initialize_session_state(self):
        if f'{self.prefix}_generated_tweets' not in st.session_state:
            st.session_state[f'{self.prefix}_generated_tweets'] = []
        if f'{self.prefix}_add_url_tweet' not in st.session_state:
            st.session_state[f'{self.prefix}_add_url_tweet'] = False
        if f'{self.prefix}_url_tweet' not in st.session_state:
            st.session_state[f'{self.prefix}_url_tweet'] = ""

    def parse_tweets(self, llm_response: str) -> List[str]:
        """Parse LLM response into a list of tweets."""
        tweets = []
        for tweet in llm_response.split('TWEET:')[1:]:
            clean_tweet = tweet.strip().split('---')[0].strip()
            if clean_tweet:
                tweets.append(clean_tweet)
        return tweets

    def generate_tweet_thread(self, config: Dict[str, Any], product: Dict[str, Any]) -> List[str]:
        """Generate a tweet thread based on a product's content."""
        prompt = config['promotetwitter']['randomtweet_prompt'].format(
            title=product['title'],
            content=product['content'],
            keywords=product['keywords']
        )
        llm_response = self.process_with_llm(
            prompt,
            config.get('llm', {}).get('llm_sys_prompt', ''),
            product['content']
        )
        return self.parse_tweets(llm_response)

    def post_tweet_thread(self, config: Dict[str, Any], tweets: List[str]):
        """Post the tweet thread to Twitter."""
        twitter_api = TwitterAPI(self.plugin_manager.config)
        try:
            twitter_api.create_thread(tweets)
            st.success(t("randomtweet_success"))
        except Exception as e:
            st.error(t("randomtweet_error") + str(e))

    def display(self, config: Dict[str, Any]):
        st.header(t("randomtweet_header"))

        # Use ProductSelectorWidget to select a product
        product_selector = ProductSelectorWidget("product_selector", f"{self.prefix}_product_selector", self.plugin_manager)
        selected_product = product_selector.display()

        if not selected_product:
            st.warning("No product selected. Please select a product to continue.")
            return

        # Display product information
        st.write(f"**Selected Product**: {selected_product['title']}")
        st.write(f"**Keywords**: {selected_product['keywords']}")
        st.markdown(f"**Content**: {selected_product['content'][:1000]}{'...' if len(selected_product['content']) > 1000 else ''}")

        # LLM Prompt
        prompt = st.text_area(
            t("randomtweet_prompt"),
            value=config['promotetwitter']['randomtweet_prompt'],
            key=f"{self.prefix}_prompt",
            height=150
        )
        config['promotetwitter']['randomtweet_prompt'] = prompt

        # Generate tweet thread button
        if st.button(t("randomtweet_generate"), key=f"{self.prefix}_generate"):
            with st.spinner(t("randomtweet_generating")):
                tweets = self.generate_tweet_thread(config, selected_product)
                st.session_state[f'{self.prefix}_generated_tweets'] = tweets
                st.session_state[f'{self.prefix}_add_url_tweet'] = False
                st.session_state[f'{self.prefix}_url_tweet'] = f"Retrouvez plus sur {selected_product['url']}" if selected_product['url'] else ""

        # Display generated tweets
        if st.session_state[f'{self.prefix}_generated_tweets']:
            st.subheader(t("randomtweet_preview"))
            for i, tweet in enumerate(st.session_state[f'{self.prefix}_generated_tweets']):
                edited_tweet = st.text_area(
                    f"Tweet {i+1}",
                    tweet,
                    key=f"{self.prefix}_tweet_{i}",
                    height=100
                )
                st.session_state[f'{self.prefix}_generated_tweets'][i] = edited_tweet

                # Validate tweet length
                if len(edited_tweet) > 280:
                    st.warning(
                        f"⚠️ Tweet {i+1} exceeds 280 characters ({len(edited_tweet)} characters). Please shorten it."
                    )

            # Checkbox for adding URL tweet (shown only after tweets are generated)
            if selected_product['url']:
                add_url_tweet = st.checkbox(
                    t("randomtweet_add_url"),
                    value=st.session_state[f'{self.prefix}_add_url_tweet'],
                    key=f"{self.prefix}_add_url"
                )
                if add_url_tweet != st.session_state[f'{self.prefix}_add_url_tweet']:
                    st.session_state[f'{self.prefix}_add_url_tweet'] = add_url_tweet
                    if not add_url_tweet:
                        st.session_state[f'{self.prefix}_url_tweet'] = ""  # Reset URL tweet if unchecked
                    st.rerun()

                # Display and edit URL tweet if checkbox is checked
                if add_url_tweet:
                    url_tweet = st.text_area(
                        "URL Tweet",
                        value=st.session_state[f'{self.prefix}_url_tweet'],
                        key=f"{self.prefix}_url_tweet",
                        height=100
                    )

                    # Validate URL tweet length
                    if len(url_tweet) > 280:
                        st.warning(
                            f"⚠️ URL Tweet exceeds 280 characters ({len(url_tweet)} characters). Please shorten it."
                        )

        # Post tweet thread button
        if st.session_state[f'{self.prefix}_generated_tweets']:
            if st.button(t("randomtweet_post"), key=f"{self.prefix}_post"):
                with st.spinner(t("randomtweet_posting")):
                    # Prepare final tweet thread
                    final_tweets = st.session_state[f'{self.prefix}_generated_tweets'].copy()
                    if st.session_state[f'{self.prefix}_add_url_tweet'] and st.session_state[f'{self.prefix}_url_tweet']:
                        final_tweets.append(st.session_state[f'{self.prefix}_url_tweet'])
                    self.post_tweet_thread(config, final_tweets)
