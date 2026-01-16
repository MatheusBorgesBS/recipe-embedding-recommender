# recipe-embedding-recommender

This repository contains a recipe recommendation system that suggests recipes to users based on the ingredients they have. The system leverages Word2Vec embeddings to understand the semantic relationships between ingredients and recipes.

## How It Works

The recommendation engine is built through a multi-step process detailed in the `chatbot_receitas.ipynb` notebook:

1.  **Data Loading & Pre-processing:**
    *   Recipes are loaded from a collection of JSON files.
    *   Ingredient lists and recipe titles are cleaned using regular expressions to remove quantities, units (e.g., "cup", "tablespoon"), advertisements, and special characters.

2.  **Word Embedding with Word2Vec:**
    *   A Word2Vec (Skip-gram) model is trained on the corpus of cleaned recipe titles and ingredients. This creates a vector space where words with similar culinary contexts (e.g., 'butter' and 'oil', or 'chicken' and 'pheasant') are positioned closely together.

3.  **Recipe Vectorization:**
    *   Each recipe is transformed into a single numerical vector. This is achieved by calculating the mean of the Word2Vec vectors for all of its 'clean' ingredients. This vector represents the overall semantic profile of the recipe.

4.  **Recommendation Logic:**
    *   When a user inputs a list of ingredients, a `query_vector` is created by averaging the embeddings of the provided ingredients.
    *   A hybrid scoring model is used to find the best recipe:
        *   **Ingredient Match Score (70% weight):** This score is a direct ratio of how many of the user's ingredients are present in a recipe.
        *   **Semantic Similarity Score (30% weight):** This is the cosine similarity between the user's `query_vector` and each recipe's vector. It helps find recipes that are conceptually similar, even if they don't contain the exact ingredients.
    *   The recipe with the highest final combined score is selected as the top recommendation.

## Files

*   **`chatbot_receitas.ipynb`**: A Jupyter Notebook detailing the entire process, including data loading, EDA, text pre-processing, Word2Vec model training, and the development of the recommendation logic.
*   **`chatbot.py`**: A command-line chatbot script that loads the pre-trained model and interacts with the user. It takes ingredient inputs and returns a full recipe recommendation.
*   **`recipe_chatbot.pkl`**: A serialized Python object (pickle file) containing the necessary artifacts for the chatbot to function. This includes the trained Word2Vec model, the pre-calculated vectors for all recipes, the recipe data, and the ingredient vocabulary.
*   **`recipes_raw.zip`**: The raw dataset, a zip archive containing three JSON files with a total of 125,164 recipes.

## How to Use

To interact with the chatbot, run the `chatbot.py` script from your terminal.

```bash
python chatbot.py
```

The script will prompt you for input. Enter the ingredients you have, and the bot will respond with the best recipe it finds.

**Example Interaction:**

```
You: I have chicken, butter, onion
Bot:
the beter recipe for you is:

Roast Meat Loaf or "Hedgehog"

Ingredients:
- 8 ounces mushrooms, sliced
- Butter
- Salt
- Freshly ground black pepper
- Grated nutmeg
- 8 ounces chicken livers
- 1 pound each ground beef, ground pork, ground veal
- 1 pound sausage meat
- 1 large onion, grated
- 3 fat cloves garlic, crushed to a paste
- 10 juniper berries, crushed
- 1 teaspoon ground allspice
- Fresh thyme sprigs
- 1 to 2 eggs
- 8 ounces unsmoked bacon
- Bay leaves
- Branches of fresh rosemary
- Tomato sauce, for serving

How to cook:
Preheat the oven to 450 degrees F.
Saute the mushrooms in butter until the juices run, then season with salt and pepper and nutmeg. Reserve.
Remove the sinews from the livers and slice. In a large bowl combine all the ground meats, sausage, livers, onion, garlic, and juniper berries. Add the allspice and thyme leaves. Season with salt and pepper. Beat the eggs and add to the mixture together with the mushrooms. Mix with your hands thoroughly.
Oil a roasting pan and place all the mixture in it, molding it into an oval shape. Adorn with the bacon slices, criss-crossed, tucking the ends under the meat. Strew some bay leaves and branches of rosemary on the top and sides. Cook in the preheated oven for 15 minutes, then lower the heat to 350 degrees and cook for 1 1/2 hours.
When cooked, there will be lots of lovely juices in the bottom of the pan. Save them to flavor soup, stocks, or eggs. Remove the loaf and place on a serving dish. Serve hot or cold with a tomato sauce.
For children, you might like to turn it into a hedgehog by pressing almonds into the meat to create prickles and placing 3 olives to form eyes and a nose. This should be done before the meat goes into the oven.
----------------------------------------
You: exit
