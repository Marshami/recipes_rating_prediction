# Modeling Recipe Ratings Using Recipe Features: An Exploration of Predictive Factors

by Danny Inoue (dinoue@ucsd.edu)

---

## Step 1: Introduction

**About:**

There are two main datasets: `recipes_df`, which contains detailed information on recipes including their names, cooking times, ingredients, steps, and nutritional information; and `interactions_df`, which records user interactions with these recipes, such as ratings and reviews. Together, these datasets enable an analysis of how various recipe attributes relate to user ratings.

#### Question to Investigate Further: What types of recipes tend to have higher average ratings?

**Reasoning:** 

Understanding which recipe attributes contribute to higher ratings can provide valuable insights for both recipe creators and users looking for quality recipes. By analyzing factors such as:

• Cooking Time (`minutes`): Do users prefer recipes that are quick to prepare?

• Number of Ingredients (`n_ingredients`): Is there a preference for simpler recipes with fewer ingredients?

• Nutritional Content (`nutrition`): Are healthier recipes receiving better ratings?

• Tags (`tags`): Do certain cuisines or dietary categories (e.g., vegetarian, gluten-free) have higher ratings?

• Number of Steps (`n_steps`): Does recipe complexity affect user satisfaction?

**Approach:**

• Data Cleaning: Ensure that all necessary data is in the correct format for analysis (e.g., converting nutrition lists to individual columns).

• Data Merging: Combine recipes_df and interactions_df on the recipe ID to associate ratings with recipe attributes.

• Exploratory Data Analysis:
Calculate the average rating for each recipe.
Analyze the distribution of ratings across different recipe attributes.

• Statistical Analysis:
Use correlation and regression analysis to identify significant relationships between recipe attributes and average ratings.

• Visualization:
Create charts and graphs to illustrate findings (e.g., scatter plots, bar charts).

**Expected Outcomes:**

By the end of this investigation, I aim to identify key factors that contribute to higher recipe ratings.

This could help in:

• Guiding recipe creators to focus on attributes that are more likely to result in higher user satisfaction.

• Assisting users in selecting recipes that are generally well-received based on their preferences.

---

## Step 2: Data Cleaning and Exploratory Data Analysis

### Step: Data Cleaning

Check for missing values in both datasets.

```py
# Load the datasets
recipes_df = pd.read_csv('RAW_recipes.csv')
interactions_df = pd.read_csv('RAW_interactions.csv')

# Display the first few rows of recipes_df
print(recipes_df.head().to_markdown(index=False))
```

| name                                 |     id |   minutes |   contributor_id | submitted   | tags                                                                                                                                                                                                                                                                                               | nutrition                                     |   n_steps | steps                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | description                                                                                                                                                                                                                                                                                                                                                                       | ingredients                                                                                                                                                                                                                             |   n_ingredients |
|:-------------------------------------|-------:|----------:|-----------------:|:------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------|----------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|
| 1 brownies in the world    best ever | 333281 |        40 |           985201 | 2008-10-27  | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings']                                                                        | [138.4, 10.0, 50.0, 3.0, 3.0, 19.0, 6.0]      |        10 | ['heat the oven to 350f and arrange the rack in the middle', 'line an 8-by-8-inch glass baking dish with aluminum foil', 'combine chocolate and butter in a medium saucepan and cook over medium-low heat , stirring frequently , until evenly melted', 'remove from heat and let cool to room temperature', 'combine eggs , sugar , cocoa powder , vanilla extract , espresso , and salt in a large bowl and briefly stir until just evenly incorporated', 'add cooled chocolate and mix until uniform in color', 'add flour and stir until just incorporated', 'transfer batter to the prepared baking dish', 'bake until a tester inserted in the center of the brownies comes out clean , about 25 to 30 minutes', 'remove from the oven and cool completely before cutting']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | these are the most; chocolatey, moist, rich, dense, fudgy, delicious brownies that you'll ever make.....sereiously! there's no doubt that these will be your fav brownies ever for you can add things to them or make them plain.....either way they're pure heaven!                                                                                                              | ['bittersweet chocolate', 'unsalted butter', 'eggs', 'granulated sugar', 'unsweetened cocoa powder', 'vanilla extract', 'brewed espresso', 'kosher salt', 'all-purpose flour']                                                          |               9 |
| 1 in canada chocolate chip cookies   | 453467 |        45 |          1848091 | 2011-04-11  | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings']                                                                                                                                      | [595.1, 46.0, 211.0, 22.0, 13.0, 51.0, 26.0]  |        12 | ['pre-heat oven the 350 degrees f', 'in a mixing bowl , sift together the flours and baking powder', 'set aside', 'in another mixing bowl , blend together the sugars , margarine , and salt until light and fluffy', 'add the eggs , water , and vanilla to the margarine / sugar mixture and mix together until well combined', 'add in the flour mixture to the wet ingredients and blend until combined', 'scrape down the sides of the bowl and add the chocolate chips', 'mix until combined', 'scrape down the sides to the bowl again', 'using an ice cream scoop , scoop evenly rounded balls of dough and place of cookie sheet about 1 - 2 inches apart to allow for spreading during baking', 'bake for 10 - 15 minutes or until golden brown on the outside and soft & chewy in the center', 'serve hot and enjoy !']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | this is the recipe that we use at my school cafeteria for chocolate chip cookies. they must be the best chocolate chip cookies i have ever had! if you don't have margarine or don't like it, then just use butter (softened) instead.                                                                                                                                            | ['white sugar', 'brown sugar', 'salt', 'margarine', 'eggs', 'vanilla', 'water', 'all-purpose flour', 'whole wheat flour', 'baking soda', 'chocolate chips']                                                                             |              11 |
| 412 broccoli casserole               | 306168 |        40 |            50969 | 2008-05-30  | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                                                                                               | [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0]     |         6 | ['preheat oven to 350 degrees', 'spray a 2 quart baking dish with cooking spray , set aside', 'in a large bowl mix together broccoli , soup , one cup of cheese , garlic powder , pepper , salt , milk , 1 cup of french onions , and soy sauce', 'pour into baking dish , sprinkle remaining cheese over top', 'bake for 25 minutes or until cheese is lightly browned', 'sprinkle with rest of french fried onions and bake until onions are browned and cheese is bubbly , about 10 more minutes']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one  #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions']                                                                   |               9 |
| millionaire pound cake               | 286009 |       120 |           461724 | 2008-02-12  | ['time-to-make', 'course', 'cuisine', 'preparation', 'occasion', 'north-american', 'desserts', 'american', 'southern-united-states', 'dinner-party', 'holiday-event', 'cakes', 'dietary', 'christmas', 'thanksgiving', 'low-sodium', 'low-in-something', 'taste-mood', 'sweet', '4-hours-or-less'] | [878.3, 63.0, 326.0, 13.0, 20.0, 123.0, 39.0] |         7 | ['freheat the oven to 300 degrees', 'grease a 10-inch tube pan with butter , dust the bottom and sides with flour , and set aside', 'in a large mixing bowl , cream the butter and sugar with an electric mixer and add the eggs one at a time , beating after each addition', 'alternately add the flour and milk , stirring till the batter is smooth', 'add the two extracts and stir till well blended', 'scrape the batter into the prepared pan and bake till a cake tester or knife blade inserted in the center comes out clean , about 1 1 / 2 hours', 'cool the cake in the pan on a rack for 5 minutes , then turn it out on the rack to cool completely']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | why a millionaire pound cake?  because it's super rich!  this scrumptious cake is the pride of an elderly belle from jackson, mississippi.  the recipe comes from "the glory of southern cooking" by james villas.                                                                                                                                                                | ['butter', 'sugar', 'eggs', 'all-purpose flour', 'whole milk', 'pure vanilla extract', 'almond extract']                                                                                                                                |               7 |
| 2000 meatloaf                        | 475785 |        90 |          2202916 | 2012-03-06  | ['time-to-make', 'course', 'main-ingredient', 'preparation', 'main-dish', 'potatoes', 'vegetables', '4-hours-or-less', 'meatloaf', 'simply-potatoes2']                                                                                                                                             | [267.0, 30.0, 12.0, 12.0, 29.0, 48.0, 2.0]    |        17 | ['pan fry bacon , and set aside on a paper towel to absorb excess grease', 'mince yellow onion , red bell pepper , and add to your mixing bowl', 'chop garlic and set aside', 'put 1tbsp olive oil into a saut pan , along with chopped garlic , teaspoons white pepper and a pinch of kosher salt', 'bring to a medium heat to sweat your garlic', 'preheat oven to 350f', 'coarsely chop your baby spinach add to your heated pan , stir frequently for approximately 5 min to wilt', 'add your spinach to the mixing bowl', 'chop your now cooled bacon , and add it to the mixing bowl', 'add your meatloaf mix to the bowl , with one egg and mix till thoroughly combined', 'add your goat cheese , one egg , 1 / 8 tsp white pepper and 1 / 8 tsp of kosher salt and mix till thoroughly combined', 'transfer to a 9x5 meatloaf pan , and cook for 60 min or until the internal temperature is at least 160f', 'let stand for 5min', 'melt 1tbsp unsalted butter into a frying pan , and cook up to three eggs at a time', 'crack each egg into a separate dish , in order to prevent egg shells from reaching the pan , then add salt and pepper to taste', 'wait until the egg whites are firm looking , but slightly runny on top before flipping your eggs', 'after flipping , wait 10~20 seconds before removing each egg and placing it over your slices of meatloaf'] | ready, set, cook! special edition contest entry: a mediterranean flavor inspired meatloaf dish. featuring: simply potatoes - shredded hash browns, egg, bacon, spinach, red bell pepper, and goat cheese.                                                                                                                                                                         | ['meatloaf mixture', 'unsmoked bacon', 'goat cheese', 'unsalted butter', 'eggs', 'baby spinach', 'yellow onion', 'red bell pepper', 'simply potatoes shredded hash browns', 'fresh garlic', 'kosher salt', 'white pepper', 'olive oil'] |              13 |

Convert `submitted` and `date` columns to datetime.

Convert `rating` to `numeric` (ensure it).

The `nutrition` column in `recipes_df` contains a list of nutritional values. Split this into separate columns for easier analysis.

Parse the `tags` and `steps` Columns. These columns contain string representations of lists. Convert them to actual lists.

Rename `id` to `recipe_id` in `recipes_df` to match `interactions_df`.

Merge the datasets on `recipe_id`.

Now, `merged_df` contains all necessary information for analysis.

In `merged_df`, fill all ratings of 0 with np.nan.

Find the average rating per recipe, as a Series.

Add this Series containing the `average_ratings` back to the `recipes` dataset.

#### Explanation:

**Resetting Index:**

• We reset the index of average_ratings so that recipe_id becomes a column instead of the index. This is necessary for merging.

**Merging Datasets:**

• We perform a **left merge** of `recipes_df` with `average_ratings` on the `recipe_id` column.

• The `how='left'` parameter ensures that all recipes in `recipes_df` are retained, even if they don't have an average rating (e.g., if they have no ratings in `interactions_df`).

**Renaming Columns:**

• We rename the `rating` column to `average_rating` to distinguish it from individual ratings and clarify that it represents the average.

### Step: Univariate Analysis

Explore the distributions of key variables separately.

**Observation:**
This bar chart shows the frequency of each rating value (e.g., from 0 to 5).
We can observe how users rate recipes on average.

**Observation:**
The histogram displays how recipe preparation times are distributed.
We can see whether most recipes are quick to prepare or take longer.

**Observation:**
This histogram shows the complexity of recipes based on the number of steps.
It helps us understand whether recipes tend to be simple or complex.

### Step: Bivariate Analysis

Examine relationships between pairs of variables to identify possible associations.

**Observation:**
This scatter plot helps identify if there's a correlation between preparation time and average rating.
We can look for trends, such as whether quicker recipes tend to have higher ratings.

**Observation:**
This plot examines if the complexity of a recipe (as measured by the number of steps) affects its average rating.
We can see if simpler recipes are rated higher.

### Step: Interesting Aggregates

We'll explore aggregate statistics by grouping and pivoting data.

**Observation:**
We can see which tags are associated with higher-rated recipes. This helps identify popular cuisines or recipe categories.

**Observation:**
This plot shows how the average caloric content of recipes varies with the number of steps.
It can indicate whether more complex recipes tend to be higher or lower in calories.

---

## Step 3: Assessment of Missingness

### Step: NMAR Analysis

In this section, we'll explore whether any missing data in our dataset is **Not Missing At Random (NMAR)**. Recall that NMAR occurs when the probability of missingness is related to the missing values themselves, and not solely to observed data.

The `description` column in the `recipes_df` dataset has a number of missing values.

We need to consider whether the missingness in the description column depends on the missing values themselves.

**Hypothesis:**

• Scenario 1 (NMAR): Recipe authors might omit the description because they believe the description is not necessary due to the simplicity or obviousness of the recipe. Alternatively, they might intentionally leave it blank if they have nothing special to mention about the recipe. In this case, the missingness depends on the content that would have been in the description (i.e., the description itself).

• Scenario 2 (Not NMAR): The missingness could be due to other factors, such as the experience level of the contributor, the time when the recipe was submitted, or whether the recipe is a variation of a common dish.

**Conclusion:**

• Without additional data or domain knowledge, it's plausible that the missingness in `description` is NMAR because the absence of a description might be related to the content that the contributor chose not to provide.

• For example, if a recipe is extremely simple (e.g., boiling eggs), the contributor might skip the description, thinking it's unnecessary. In this case, the missingness depends on the nature of the recipe itself, which is unobserved in the `description` field.

**Hypothesis:**

• Scenario 1 (NMAR): Users might skip writing a review if they had an extremely negative or extremely positive experience that they don't wish to articulate. Alternatively, they might skip the review if they have privacy concerns or time constraints. If the decision to omit the review depends on their feelings about the recipe (which are unobserved), the missingness is NMAR.

• Scenario 2 (Not NMAR): The missingness could depend on other observed factors, such as the rating given (e.g., users who rate 5 stars might be more likely to leave a review).

**Conclusion:**

• It's possible that the missingness in `review` is NMAR because the absence of a review might be directly related to the user's unrecorded opinions or experiences with the recipe.

• For instance, a user who found the recipe perfect might not feel the need to elaborate further, believing the 5-star rating suffices. In this case, the missingness in the `review` column depends on the user's unobserved feelings or experiences with the recipe.

### Step: Missingness Dependency

Analyze the missingness of a selected column and test whether it depends on other columns in the dataset. Use permutation tests to statistically assess these dependencies.

#### Selected Column with Missingness: `description` in `recipes_df`

As previously identified, the `description` column has non-trivial missingness.

**Objective:**

• Find at least one column that the missingness of description depends on.

• Find at least one column that the missingness of description does not depend on.

**Analyzing Dependency on** `n_steps`

**Hypothesis:** 
The missingness of `description` depends on the complexity of the recipe, which can be approximated by the number of steps (`n_steps`). Recipes with fewer steps might be less likely to have a description.

**Permutation Test:**
We'll perform a permutation test to assess whether the missingness of `description` depends on `n_steps`.

**Interpretation:**

• If the p-value is small (e.g., less than 0.05), we reject the null hypothesis that the missingness of `description` is independent of `n_steps`. This suggests that there is a significant dependency.

**Result:**

**Low Missingness in** `description` **:**

• The description column has a very low proportion of missing values (only 0.08%).

• Such a small amount of missing data may not provide sufficient statistical power to detect dependencies through permutation tests.

**Permutation Test Indicates No Significant Dependency:**

• The p-value is greater than the conventional significance level of 0.05.

• Conclusion: We **fail to reject the null hypothesis** that the missingness of `description` is independent of `n_steps`.

• Interpretation: **There is insufficient evidence to suggest that the missingness of the** `description` **column depends on the number of steps in a recipe.**

**Observed Difference in Means:**

• The observed difference in mean `n_steps` between recipes with and without descriptions is approximately −1.

• This suggests that recipes with missing descriptions have, on average, one more step than those with descriptions.

• However, this difference is not statistically significant based on the permutation test.

**Possible Reasons for the Results:**

**Insufficient Data for Missingness Analysis:**

• With only 70 missing descriptions, the sample size for the group with missing data is very small.

• Small sample sizes can lead to high variability and reduced statistical power.

**Random Missingness:**

• The missingness might be **Missing Completely At Random (MCAR)**.

• Since the missingness does not appear to depend on n_steps or potentially other variables, it may be randomly distributed.

#### Testing Dependency Between Missingness of `description` and `submitted` Date

**Hypotheses:**

• **Null Hypothesis**: The missingness of the `description` column is **independent** of the `submitted` date. Any observed difference is due to random chance.

• **Alternative Hypothesis**: The missingness of the `description` column **depends** on the `submitted` date. Recipes submitted in certain time periods are more likely to have missing descriptions.

**Data Preparation:**

**Visualize Missingness Over Time:**

**Interpretation:**

• If the proportion of missing descriptions varies significantly over the years, it suggests that missingness depends on the submission year.

**Defining Groups Based on Submission Year:**

To perform a statistical test, we'll divide the data into two groups:

• Group A: Recipes submitted before a certain year (e.g., 2010).

• Group B: Recipes submitted in or after that year.

**Calculating the Observed Difference:**

**Performing the Permutation Test:**

**Interpretation of Results:**

**Observed Difference:**

• Proportion Missing (Submitted Before 2010): This represents the proportion of recipes submitted before 2010 that have missing descriptions.

• Proportion Missing (Submitted In/After 2010): This represents the proportion of recipes submitted in or after 2010 that have missing descriptions.

• Observed Difference: The difference between the two proportions.

**P-value Interpretation**

• P-value < 0.05: Reject the null hypothesis. There is evidence that the missingness of description depends on the submission date.

• P-value ≥ 0.05: Fail to reject the null hypothesis. There is insufficient evidence to suggest dependency.

Since the p-value is less than the significance level of 0.05, we **reject the null hypothesis**. This provides strong statistical evidence in favor of the alternative hypothesis.

**Conclusion:**

There is significant evidence to suggest that the missingness of the `description` column **depends** on the submission date of the recipes. Specifically, recipes submitted during certain time periods are more likely to have missing descriptions than others.

The missingness of the description column is likely **Missing At Random (MAR)** rather than **Missing Completely At Random (MCAR)**. Since missingness depends on an observed variable (submission date), standard analyses that assume MCAR may not be appropriate.

---

## Step 4: Hypothesis Testing

In this step, we will perform a hypothesis test to determine whether there is a significant difference in average ratings between recipes that take **30 minutes or less** to prepare and those that take **more than 30 minutes**.

#### Null and Alternative Hypotheses:

• **Null Hypothesis**: There is **no difference** in the average rating between recipes that take **30 minutes or less** and recipes that take **more than 30 minutes** to prepare.

• **Alternative Hypothesis**: There is a **difference** in the average rating between recipes that take **30 minutes or less** and recipes that take **more than 30 minutes** to prepare.

#### Test Statistic:

• We will use the **difference in sample means** between the two groups as our test statistic.

#### Significance Level:

• We will use a significance level of **0.05**.

### Step: Data Preparation

Ensure that we have a dataset that includes both recipe information and user ratings.

### Step: Calculate Observed Test Statistic

Compute the mean ratings for each group and calculate the observed difference.

#### Permutation Test:

Perform a permutation test to assess whether the observed difference is statistically significant.

**Steps:**

• Combine All Ratings: Under the null hypothesis, the grouping is arbitrary.

• Shuffle Time Categories: Randomly assign 'short' and 'long' labels to the ratings.

• Recalculate the Difference in Means: For each permutation, compute the difference in mean ratings.

• Repeat: Perform this process multiple times to build a distribution under the null hypothesis.

• Calculate P-value: Determine the proportion of permutations where the absolute permuted difference is greater than or equal to the observed difference.

**Implementation:**

**Visualization:** Plot the distribution of permuted differences and indicate the observed difference.

**Conclusion:**

This suggests that there is a **statistically significant difference** in the average ratings between recipes that take **30 minutes or less** and those that take **more than 30 minutes** to prepare.

**Justification:**

• Choice of Test: We used a permutation test because it makes no assumptions about the distribution of ratings and is appropriate for comparing the means of two independent groups.

• Test Statistic: The difference in sample means is a straightforward and interpretable measure for comparing group averages.

• Significance Level: A 5% significance level is standard in hypothesis testing and balances the risk of Type I and Type II errors.

**Interpretation:**

• Our findings suggest that preparation time is associated with differences in recipe ratings on the platform. Users may have preferences influenced by how long a recipe takes to prepare, which could affect their overall satisfaction and rating of the recipe.

**Note:**

• While the statistical test indicates a significant difference, it does not establish causation. Other factors not accounted for in this analysis may also influence recipe ratings.

---

## Step 5: Framing a Prediction Problem

#### Problem Identification:

We will identify a prediction problem based on our dataset and previous analyses. The goal is to develop a predictive model that estimates a specific outcome using features available in the data.

**Proposed Prediction Problem:** Predicting the Average Rating of Recipes

**Type of Problem:** Regression

#### Justification:

**Relevance to Previous Analysis:**

• In Steps 1-4, we explored factors that might influence recipe ratings, such as preparation time (`minutes`), number of steps (`n_steps`), number of ingredients (`n_ingredients`), and nutritional content.

• Building a model to predict the average rating of recipes aligns with our earlier analyses and maintains a coherent theme throughout the project.

**Practical Importance:**

• For Recipe Creators: Understanding which attributes contribute to higher ratings can help in designing recipes that are more appealing to users.

• For Users: A predictive model can aid in recommending recipes that are likely to be well-received based on their preferences.

**Data Availability:**

We have access to a rich set of features from the recipes_df and interactions_df datasets, including:

• Quantitative Features: Preparation time, number of steps, number of ingredients, nutritional information (calories, protein, fat, etc.).

• Categorical Features: Tags (e.g., cuisine type, dietary preferences), ingredient lists.

• Textual Data: Descriptions and steps (optional, if we choose to incorporate natural language processing techniques).

**Feasibility:**

There is sufficient data to train and validate a regression model.
The target variable (rating) is continuous, making it suitable for regression analysis.

---

## Step 6: Baseline Model

In this step, we'll train a baseline regression model to predict the average rating of recipes using at least two features from our dataset. We'll implement all data preprocessing and model training steps in a single scikit-learn Pipeline.

First, we'll import the required libraries.

### Step: Data Preparation

Assuming that the datasets `recipes_df` and `interactions_df` have been loaded and preprocessed as per previous steps.

### Step: Feature Selection

We'll select at least two features to use in our baseline model:

• **Numerical Features:** `minutes`, `n_ingredients`, `n_steps`, `calories`, `protein_DV`, `carbs_DV`

Our target variable will be `average_rating`.

### Step: Prepare the Data

Prepare the features (`X`) and the target variable (`y`), and check for missing values.

### Step: Split Data into Training and Testing Sets

Split the data into training and testing sets to evaluate the model's ability to generalize to unseen data.

### Step: Create a Pipeline

Create a scikit-learn Pipeline that includes data preprocessing (scaling) and model training.

### Step: Train the Baseline Model

### Step: Evaluate the Model

Evaluate the model's performance on the test set.

#### Interpretation of Results:

• The low R-squared value suggests that the model is not capturing much of the variance in the data. This indicates that our numerical features alone may not be strong predictors of average recipe ratings.

• The baseline model provides a starting point for our predictive task. We can aim to improve performance by incorporating additional features and exploring more complex models in the next steps.

---

## Step 7: Final Model

In this step, we will build a **final model** that improves upon the baseline model we created earlier. 

We will achieve this by:

• **Engineering at least two new features** from the data.

• Implementing all steps in a single **scikit-learn Pipeline**.

• **Performing hyperparameter tuning** to optimize our model's performance.

• Using the **same training and testing datasets** as in the baseline model to ensure a fair comparison.

### Recap of the Baseline Model

Our baseline model used a Linear Regression model with two features:

• `minutes`

• `n_ingredients`

The performance of the baseline model was limited, with low R-squared scores indicating that the model did not capture much variance in the target variable (`rating`).

### Objective

Our goal is to improve the predictive performance by:

• **Engineering new features** that may have a stronger relationship with the target variable.

• **Including categorical variables** by encoding them appropriately.

• **Using a more advanced regression algorithm** that can capture complex patterns.

• **Tuning hyperparameters** to find the optimal model settings.

### Feature Engineering

We will engineer at least two new features:

**Feature 1:** `minutes_per_step`

• Definition: Average time spent per step in the recipe.

• Formula: `minutes_per_step = minutes / n_steps`

• Rationale: This feature captures the complexity or intensity of each step. A lower value may indicate quick, simple steps, while a higher value may indicate more time-consuming steps.

**Feature 2:** `ingredients_per_step`

• Definition: Average number of ingredients used per step.

• Formula: `ingredients_per_step = n_ingredients / n_steps`

• Rationale: This feature reflects the complexity of each step in terms of ingredients used. It may help in understanding how ingredient-heavy each step is.

**Feature 3:** Encoding `tags`

• We will encode the tags column to include categorical information about each recipe.

• Approach: Use one-hot encoding for the most frequent tags.

• Rationale: Tags provide valuable categorical information that may influence ratings (e.g., dietary preferences, cuisine types).

### Data Preparation

We will continue to predict individual `ratings` from `interactions_df`, merged with recipe features from `recipes_df`.

#### Handle Infinite or NaN Values:

• Recipes with `n_steps` equal to zero will result in division by zero.

• Replace infinite values with zero or appropriate value.

### Encoding Categorical Variables

#### Select Most Frequent Tags

• Select the top 20 most frequent tags to reduce dimensionality.

#### Create Dummy Variables for Top Tags

#### Feature Selection

**Our final set of features (X) includes:**

• **Numerical Features:** `minutes`, `n_ingredients`, `n_steps`, `minutes_per_step`, `ingredients_per_step`

• **Categorical Features (Encoded):** Binary features for each of the top 20 tags

The target variable (`y`) is the individual `rating`.

### Split Data into Training and Testing Sets

We will use the same train-test split as in the baseline model for consistency.

#### Using RandomizedSearchCV for Hyperparameter Tuning

**RandomizedSearchCV** is a method provided by scikit-learn that allows you to perform hyperparameter tuning by sampling a fixed number of parameter settings from specified distributions. Instead of exhaustively trying all possible combinations (as in Grid Search), Randomized Search evaluates a random selection of combinations, which can significantly reduce computation time while still exploring a wide range of hyperparameters.

#### Define the Hyperparameter Distributions

Define distributions from which parameter values will be sampled.

#### Set Up RandomizedSearchCV

**Parameters:**

• `n_iter`: The number of parameter settings that are sampled. You can adjust this number based on available resources.

• `random_state`: Ensures reproducibility of results.

#### Interpretation of Results:

The final model is better than the baseline model from Step 6, as evidenced by:

• **Lower MSE and MAE**: Both error metrics have decreased, indicating that your model's predictions are, on average, closer to the actual ratings.

• **Higher R-squared Score**: The increase in R-squared suggests that your model captures slightly more of the variability in the target variable.

However, the improvements are modest. This suggests that while the additional features and hyperparameter tuning have contributed to better performance, the overall impact is limited.

#### Possible Reasons for Limited Improvement:

**Predicting Individual Ratings is Challenging:**

• Subjectivity: User ratings are highly subjective and can vary widely due to personal preferences, mood, or experiences.

• Unobserved Factors: Many factors influencing ratings may not be captured in the dataset (e.g., taste preferences, cooking skills).

**Feature Influence:**

• Weak Predictors: The features engineered may not have a strong enough relationship with the target variable to produce significant improvements.

• Data Limitations: There may be insufficient variability or signal in the features to allow the model to learn better patterns.

**Model Complexity:**

• Marginal Gains from Tuning: Hyperparameter tuning can lead to diminishing returns if the model is already capturing most of the learnable patterns from the data.

#### Conclusion:

While there was a shift from predicting **average ratings** to predicting **individual ratings**, the final model remains closely related to the prediction problem identified in **Step 5**. The change was necessitated by practical considerations, specifically to avoid data leakage and ensure a valid modeling process.

**Key Points of Alignment:**

• Objective Consistency: The core goal of predicting recipe ratings based on features is maintained.

• Methodological Continuity: The regression framework and feature set are consistent with the initial problem statement.

• Insight Generation: The model contributes to understanding how recipe characteristics influence user ratings, fulfilling the intent of Step 5.

**Implications for the Project:**

• Coherent Theme: The project maintains a coherent focus on recipe ratings throughout all steps.

• Practical Application: The final model can be used to estimate how new recipes might be rated by users, providing value to recipe creators and platforms.

• Future Work: The experience highlights the importance of considering data limitations and potential adjustments in predictive modeling.

---

## Step 8: Fairness Analysis

In this step, we will perform a **fairness analysis** of our final model from Step 7. The goal is to investigate whether our model's performance differs between two distinct groups, which could indicate potential biases or unfairness.

#### Defining the Groups
We will analyze the model's performance for recipes categorized as **"easy"** versus those that are **not categorized as "easy"**.

• **Group X ("easy" recipes):** Recipes that have the tag **"easy".**

• **Group Y ("not easy" recipes):** Recipes that do **not** have the tag **"easy".**

**Reason for Choosing These Groups:**

• The **"easy"** tag is a common descriptor that might influence user expectations and ratings.

• It's important to ensure that the model performs equally well for recipes of varying complexity levels.

• Investigating this can reveal whether the model is inadvertently biased against simpler or more complex recipes.

#### Evaluation Metric

Since our model is a **regression model** predicting user ratings, we will use the **Root Mean Squared Error (RMSE)** as our evaluation metric.

• **RMSE** measures the average magnitude of the prediction errors.

• It's appropriate for regression tasks and is sensitive to large errors, making it suitable for fairness analysis.

#### Hypotheses

• **Null Hypothesis:** Our model is fair. The RMSE of the model is the same for "easy" recipes and "not easy" recipes. Any observed difference in RMSE between the two groups is due to random chance.

• **Alternative Hypothesis:** Our model is unfair. The RMSE of the model differs between "easy" recipes and "not easy" recipes. Specifically, the model performs worse for one group compared to the other.

### Implementing the Fairness Analysis

#### Prepare the Test Data

We will use the **test set** from our final model to evaluate performance on unseen data.

#### Identify the Two Groups

#### Make Predictions for Each Group

#### Calculate the Observed RMSE for Each Group

### Permutation Test

#### Combine Residuals and Group Labels

First, we need to compute the residuals (errors) for all test samples.

#### Permutation Procedure

#### Calculate the P-value

#### Visualization

### Interpretation of the Results:

• **Statistical Significance:** 

• Our p-value is less than the conventional significance level of 0.05. 

• Since the p-value is less than 0.05, we **reject the null hypothesis**.

• **Direction of the Difference:** 

• Our result for Observed Difference in RMSE means that the RMSE for "Easy" recipes is lower than the RMSE for "Not Easy" recipes. 

• Implication: **The model performs better (i.e., has lower error) on "Easy" recipes than on "Not Easy" recipes.**

• **Fairness Implications:**

• Model Unfairness: The statistically significant difference in RMSE indicates that the model does not perform equally across the two groups.

• Potential Bias: The model is more accurate for "Easy" recipes. Users interested in "Not Easy" recipes receive less accurate predictions.

• Fairness Concern: This performance disparity suggests a potential fairness issue, as the model favors one group over the other.

#### Notes:

**Permutation Test Validity:**

• The permutation test is appropriate here as it assesses whether the observed difference in RMSE could have occurred by chance under the null hypothesis.

**Statistical Power:**

• Ensure that the sample sizes in both groups are large enough to provide sufficient statistical power.

**Limitations:**

• The fairness analysis is limited to the groups and metric chosen.

• Other forms of bias may exist that are not captured by this analysis.