from pytest_bdd import given, when, then, parsers
from plugins.common import CommonPlugin
from app import PluginManager

@given(parsers.parse('I have numbers {a:d} and {b:d}'), target_fixture='numbers')
def numbers(a, b):
    return {'a': a, 'b': b}

@when('I add them')
def add(numbers):
    mock_manager = PluginManager({'common': {}})
    plugin = CommonPlugin("common", mock_manager)
    numbers['result'] = plugin.add_numbers(numbers['a'], numbers['b'])

@then(parsers.parse('the result should be {result:d}'))
def check_result(numbers, result):
    assert numbers['result'] == result

from pytest_bdd import given, when, then, parsers
from plugins.common import CommonPlugin
from plugins.postarticle import PostarticlePlugin
from app import PluginManager, load_config
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# ... (contenu existant de steps.py, comme pour common)

@given(parsers.parse('I have a dummy post title "{title}" and content "{content}"'), target_fixture='dummy_post')
def dummy_post(title, content):
    return {
        'title': title,
        'content': content,
        'publish_immediately': False,  # Draft seulement
        'feature_image': None,  # Pas d'image pour le test
        'publication_url': None  # Utilise l'URL par défaut (première configurée)
    }

@when('I create the draft post on Substack')
def create_draft(dummy_post):
    config = load_config()  # Charge le config réel pour les tests d'intégration
    mock_manager = PluginManager(config)
    plugin = PostarticlePlugin("postarticle", mock_manager)
    # Crée le draft (utilise l'URL par défaut)
    post_result = plugin.substack_api.post(
        dummy_post['title'],
        dummy_post['content'],
        dummy_post['publish_immediately'],
        feature_image=dummy_post['feature_image'],
        publication_url=dummy_post['publication_url']
    )
    assert post_result is not None, "Failed to create draft post"
    assert post_result['status'] == 'draft', "Post was not created as draft"
    dummy_post['draft_id'] = post_result['id']

@then('the draft id should be printed for debugging')
def debug_draft_id(dummy_post):
    print(f"DEBUG: Draft ID created: {dummy_post['draft_id']}")

@then('the draft should exist in the list of drafts')
def verify_draft_exists(dummy_post):
    config = load_config()
    mock_manager = PluginManager(config)
    plugin = PostarticlePlugin("postarticle", mock_manager)
    # Liste les drafts (utilise l'URL par défaut)
    drafts = plugin.substack_api.list_drafts(publication_url=dummy_post['publication_url'])
    assert any(d['id'] == dummy_post['draft_id'] for d in drafts), f"Draft ID {dummy_post['draft_id']} not found in drafts"

@when('I delete the draft post from Substack')
def delete_draft(dummy_post):
    config = load_config()
    mock_manager = PluginManager(config)
    plugin = PostarticlePlugin("postarticle", mock_manager)
    # Supprime le draft (utilise l'URL par défaut)
    success = plugin.substack_api.delete_draft(dummy_post['draft_id'], publication_url=dummy_post['publication_url'])
    assert success, f"Failed to delete draft ID {dummy_post['draft_id']}"

@then('the draft should no longer exist in the list of drafts')
def verify_draft_deleted(dummy_post):
    config = load_config()
    mock_manager = PluginManager(config)
    plugin = PostarticlePlugin("postarticle", mock_manager)
    # Liste les drafts à nouveau
    drafts = plugin.substack_api.list_drafts(publication_url=dummy_post['publication_url'])
    assert not any(d['id'] == dummy_post['draft_id'] for d in drafts), f"Draft ID {dummy_post['draft_id']} still exists after deletion"

@then('the draft should be visible in Substack web interface')
def verify_draft_in_web_interface(dummy_post):
    config = load_config()
    publication_urls = config['common'].get('substack_publication_url', '')
    if isinstance(publication_urls, str):
        publication_urls = [url.strip() for url in publication_urls.split(';') if url.strip()]
    publication_url = publication_urls[0] if publication_urls else None
    assert publication_url, "No Substack publication URL configured"

    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    # chrome_options.add_argument("--headless")  # Décommentez pour mode headless
    service = Service("/usr/bin/chromedriver")
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Connexion à Substack
        driver.get("https://substack.com/sign-in")
        wait = WebDriverWait(driver, 20)
        email_field = wait.until(EC.presence_of_element_located((By.NAME, "email")))
        email_field.send_keys(config['common']['substack_email'])
        sign_in_link = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "Sign in with password")))
        sign_in_link.click()
        password_field = wait.until(EC.presence_of_element_located((By.NAME, "password")))
        password_field.send_keys(config['common']['substack_password'])
        password_field.send_keys(Keys.RETURN)
        time.sleep(5)  # Attendre la connexion

        # Accéder à la page des brouillons
        driver.get(f"{publication_url}/drafts")
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "post-preview-title")))
        draft_titles = driver.find_elements(By.CLASS_NAME, "post-preview-title")
        draft_found = any(dummy_post['title'] in title.text for title in draft_titles)
        assert draft_found, f"Draft with title '{dummy_post['title']}' not found in Substack web interface"

    finally:
        driver.quit()

@then('the draft should no longer be visible in Substack web interface')
def verify_draft_not_in_web_interface(dummy_post):
    config = load_config()
    publication_urls = config['common'].get('substack_publication_url', '')
    if isinstance(publication_urls, str):
        publication_urls = [url.strip() for url in publication_urls.split(';') if url.strip()]
    publication_url = publication_urls[0] if publication_urls else None
    assert publication_url, "No Substack publication URL configured"

    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    # chrome_options.add_argument("--headless")  # Décommentez pour mode headless
    service = Service("/usr/bin/chromedriver")
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Connexion à Substack
        driver.get("https://substack.com/sign-in")
        wait = WebDriverWait(driver, 20)
        email_field = wait.until(EC.presence_of_element_located((By.NAME, "email")))
        email_field.send_keys(config['common']['substack_email'])
        sign_in_link = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "Sign in with password")))
        sign_in_link.click()
        password_field = wait.until(EC.presence_of_element_located((By.NAME, "password")))
        password_field.send_keys(config['common']['substack_password'])
        password_field.send_keys(Keys.RETURN)
        time.sleep(5)  # Attendre la connexion

        # Accéder à la page des brouillons
        driver.get(f"{publication_url}/drafts")
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "post-preview-title")))
        draft_titles = driver.find_elements(By.CLASS_NAME, "post-preview-title")
        draft_found = any(dummy_post['title'] in title.text for title in draft_titles)
        assert not draft_found, f"Draft with title '{dummy_post['title']}' still visible in Substack web interface"

    finally:
        driver.quit()
