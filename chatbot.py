import pickle
import re
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(BASE_DIR, "recipe_chatbot.pkl")

with open(PKL_PATH, "rb") as f:
    artifacts = pickle.load(f)

w2v_model = artifacts["w2v_model"]
recipe_vectors = artifacts["recipe_vectors"]
recipes = artifacts["recipes"]
ingredient_vocab = artifacts["ingredient_vocab"]




stopwords = {
    'i', 'have', 'do', 'what', 'can', 'a', 'the', 'and', 'or', 'to', 'of', 'is'
}

def clean_user_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_user_input(text):
    tokens = clean_user_text(text).split()
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [t for t in tokens if t in ingredient_vocab]
    return tokens


def build_query_vector(tokens):
    vectors = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if len(vectors) == 0:
        return None
    return np.mean(vectors, axis=0)

def ingredient_match_score(user_ings, recipe_ings):
    recipe_tokens = set()

    for ing in recipe_ings:
        recipe_tokens.update(ing.split())

    if len(user_ings) == 0:
        return 0.0

    match = sum(1 for ing in user_ings if ing in recipe_tokens)
    return match / len(user_ings)

def find_best_recipe(query_vector, user_ings):
    scored = []

    for i, rv in enumerate(recipe_vectors):
        sem_sim = cosine_similarity(
            query_vector.reshape(1, -1),
            rv.reshape(1, -1)
        )[0][0]

        ing_match = ingredient_match_score(
            user_ings,
            recipes[i]['clean_ingredients']
        )

        final_score = 0.7 * ing_match + 0.3 * sem_sim
        scored.append((final_score, i))

    scored.sort(reverse=True)
    return recipes[scored[0][1]]

def chatbot_response(user_text):
    user_tokens = process_user_input(user_text)
    query_vector = build_query_vector(user_tokens)

    if query_vector is None:
        return "Não consegui identificar ingredientes conhecidos"

    best_recipe = find_best_recipe(query_vector, user_tokens)

    response = f"the beter recipe for you is:\n\n"
    response += f"{best_recipe['title']}\n\n"

    response += "Ingredients:\n"
    for ing in best_recipe['ingredients']:
        response += f"- {ing}\n"

    response += "\nHow to cook:\n"
    response += best_recipe['instructions']

    return response

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    print("\nBot:")
    print(chatbot_response(user_input))
    print("-" * 40)
