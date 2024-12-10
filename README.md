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

```py
# Load the datasets
recipes_df = pd.read_csv('RAW_recipes.csv')
interactions_df = pd.read_csv('RAW_interactions.csv')
```

Convert `submitted` and `date` columns to datetime.

```py
recipes_df['submitted'] = pd.to_datetime(recipes_df['submitted'])
interactions_df['date'] = pd.to_datetime(interactions_df['date'])
```

Convert `rating` to `numeric` (ensure it).

```py
interactions_df['rating'] = pd.to_numeric(interactions_df['rating'])
```

The `nutrition` column in `recipes_df` contains a list of nutritional values. Split this into separate columns for easier analysis.

```py
# Define nutrition column names
nutrition_cols = ['calories', 'total_fat_DV', 'sugar_DV', 'sodium_DV', 
                  'protein_DV', 'saturated_fat_DV', 'carbs_DV']

# Expand the nutrition column into separate columns
nutrition_df = recipes_df['nutrition'].str.strip('[]').str.split(',', expand=True)
nutrition_df.columns = nutrition_cols
nutrition_df = nutrition_df.apply(pd.to_numeric)

# Add the new nutrition columns to recipes_df
recipes_df = recipes_df.join(nutrition_df)

# Drop the original 'nutrition' column
recipes_df = recipes_df.drop('nutrition', axis=1)
```

Parse the `tags` and `steps` Columns. These columns contain string representations of lists. Convert them to actual lists.

```py
# Function to convert string representation of list to actual list
def str_to_list(x):
    # Check if the value is not null or empty
    if pd.isna(x) or x == '':
        return []
    else:
        # Remove square brackets
        x = x.strip('[]')
        # Split the string by commas
        items = x.split(',')
        # Strip whitespace and quotes from each item
        items = [item.strip().strip("'").strip('"') for item in items]
        # Remove any empty strings from the list
        items = [item for item in items if item]
        return items

# Apply the function to the columns
recipes_df['tags'] = recipes_df['tags'].apply(str_to_list)
recipes_df['steps'] = recipes_df['steps'].apply(str_to_list)
```

Rename `id` to `recipe_id` in `recipes_df` to match `interactions_df`.

```py
recipes_df = recipes_df.rename(columns={'id': 'recipe_id'})
```

Merge the datasets on `recipe_id`.

```py
merged_df = pd.merge(interactions_df, recipes_df, on='recipe_id', how='inner')
```

Now, `merged_df` contains all necessary information for analysis.

```py
print(merged_df.columns.tolist())
```

['user_id', 'recipe_id', 'date', 'rating', 'review', 'name', 'minutes', 'contributor_id', 'submitted', 'tags', 'n_steps', 'steps', 'description', 'ingredients', 'n_ingredients', 'calories', 'total_fat_DV', 'sugar_DV', 'sodium_DV', 'protein_DV', 'saturated_fat_DV', 'carbs_DV']

In `merged_df`, fill all ratings of 0 with np.nan.

```py
import warnings
warnings.simplefilter('ignore')

# Replace ratings of 0 with np.nan
merged_df['rating'].replace(0, np.nan, inplace=True)
merged_df['rating'].isna().any()
```

np.True_

Find the average rating per recipe, as a Series.

```py
# Calculate the average rating per recipe, ignoring NaN values
average_ratings = merged_df.groupby('recipe_id')['rating'].mean()
```

Add this Series containing the `average_ratings` back to the `recipes` dataset.

```py
# Reset index to convert 'recipe_id' from index to column
average_ratings = average_ratings.reset_index()

# Merge the average ratings with recipes_df on 'recipe_id'
recipes_df = pd.merge(recipes_df, average_ratings, on='recipe_id', how='left')

# Rename the 'rating' column to 'average_rating' for clarity
recipes_df.rename(columns={'rating': 'average_rating'}, inplace=True)
recipes_df
```

| name                                 |   recipe_id |   minutes |   contributor_id | submitted           | tags                                                                                                                                                                                                                                                                                               |   n_steps | steps                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | description                                                                                                                                                                                                                                                                                                                                                                       | ingredients                                                                                                                                                                                                                             |   n_ingredients |   calories |   total_fat_DV |   sugar_DV |   sodium_DV |   protein_DV |   saturated_fat_DV |   carbs_DV |   average_rating_x | description_missing   |   submitted_year | submitted_before_2010   |   index |   average_rating_y |
|:-------------------------------------|------------:|----------:|-----------------:|:--------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|-----------:|---------------:|-----------:|------------:|-------------:|-------------------:|-----------:|-------------------:|:----------------------|-----------------:|:------------------------|--------:|-------------------:|
| 1 brownies in the world    best ever |      333281 |        40 |           985201 | 2008-10-27 00:00:00 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings']                                                                        |        10 | ['heat the oven to 350f and arrange the rack in the middle', 'line an 8-by-8-inch glass baking dish with aluminum foil', 'combine chocolate and butter in a medium saucepan and cook over medium-low heat', 'stirring frequently', 'until evenly melted', 'remove from heat and let cool to room temperature', 'combine eggs', 'sugar', 'cocoa powder', 'vanilla extract', 'espresso', 'and salt in a large bowl and briefly stir until just evenly incorporated', 'add cooled chocolate and mix until uniform in color', 'add flour and stir until just incorporated', 'transfer batter to the prepared baking dish', 'bake until a tester inserted in the center of the brownies comes out clean', 'about 25 to 30 minutes', 'remove from the oven and cool completely before cutting']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | these are the most; chocolatey, moist, rich, dense, fudgy, delicious brownies that you'll ever make.....sereiously! there's no doubt that these will be your fav brownies ever for you can add things to them or make them plain.....either way they're pure heaven!                                                                                                              | ['bittersweet chocolate', 'unsalted butter', 'eggs', 'granulated sugar', 'unsweetened cocoa powder', 'vanilla extract', 'brewed espresso', 'kosher salt', 'all-purpose flour']                                                          |               9 |      138.4 |             10 |         50 |           3 |            3 |                 19 |          6 |                  4 | False                 |             2008 | True                    |  127796 |                  4 |
| 1 in canada chocolate chip cookies   |      453467 |        45 |          1848091 | 2011-04-11 00:00:00 | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings']                                                                                                                                      |        12 | ['pre-heat oven the 350 degrees f', 'in a mixing bowl', 'sift together the flours and baking powder', 'set aside', 'in another mixing bowl', 'blend together the sugars', 'margarine', 'and salt until light and fluffy', 'add the eggs', 'water', 'and vanilla to the margarine / sugar mixture and mix together until well combined', 'add in the flour mixture to the wet ingredients and blend until combined', 'scrape down the sides of the bowl and add the chocolate chips', 'mix until combined', 'scrape down the sides to the bowl again', 'using an ice cream scoop', 'scoop evenly rounded balls of dough and place of cookie sheet about 1 - 2 inches apart to allow for spreading during baking', 'bake for 10 - 15 minutes or until golden brown on the outside and soft & chewy in the center', 'serve hot and enjoy !']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | this is the recipe that we use at my school cafeteria for chocolate chip cookies. they must be the best chocolate chip cookies i have ever had! if you don't have margarine or don't like it, then just use butter (softened) instead.                                                                                                                                            | ['white sugar', 'brown sugar', 'salt', 'margarine', 'eggs', 'vanilla', 'water', 'all-purpose flour', 'whole wheat flour', 'baking soda', 'chocolate chips']                                                                             |              11 |      595.1 |             46 |        211 |          22 |           13 |                 51 |         26 |                  5 | False                 |             2011 | False                   |  169423 |                  5 |
| 412 broccoli casserole               |      306168 |        40 |            50969 | 2008-05-30 00:00:00 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                                                                                               |         6 | ['preheat oven to 350 degrees', 'spray a 2 quart baking dish with cooking spray', 'set aside', 'in a large bowl mix together broccoli', 'soup', 'one cup of cheese', 'garlic powder', 'pepper', 'salt', 'milk', '1 cup of french onions', 'and soy sauce', 'pour into baking dish', 'sprinkle remaining cheese over top', 'bake for 25 minutes or until cheese is lightly browned', 'sprinkle with rest of french fried onions and bake until onions are browned and cheese is bubbly', 'about 10 more minutes']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one  #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions']                                                                   |               9 |      194.8 |             20 |          6 |          32 |           22 |                 36 |          3 |                  5 | False                 |             2008 | True                    |  116293 |                  5 |
| millionaire pound cake               |      286009 |       120 |           461724 | 2008-02-12 00:00:00 | ['time-to-make', 'course', 'cuisine', 'preparation', 'occasion', 'north-american', 'desserts', 'american', 'southern-united-states', 'dinner-party', 'holiday-event', 'cakes', 'dietary', 'christmas', 'thanksgiving', 'low-sodium', 'low-in-something', 'taste-mood', 'sweet', '4-hours-or-less'] |         7 | ['freheat the oven to 300 degrees', 'grease a 10-inch tube pan with butter', 'dust the bottom and sides with flour', 'and set aside', 'in a large mixing bowl', 'cream the butter and sugar with an electric mixer and add the eggs one at a time', 'beating after each addition', 'alternately add the flour and milk', 'stirring till the batter is smooth', 'add the two extracts and stir till well blended', 'scrape the batter into the prepared pan and bake till a cake tester or knife blade inserted in the center comes out clean', 'about 1 1 / 2 hours', 'cool the cake in the pan on a rack for 5 minutes', 'then turn it out on the rack to cool completely']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | why a millionaire pound cake?  because it's super rich!  this scrumptious cake is the pride of an elderly belle from jackson, mississippi.  the recipe comes from "the glory of southern cooking" by james villas.                                                                                                                                                                | ['butter', 'sugar', 'eggs', 'all-purpose flour', 'whole milk', 'pure vanilla extract', 'almond extract']                                                                                                                                |               7 |      878.3 |             63 |        326 |          13 |           20 |                123 |         39 |                  5 | False                 |             2008 | True                    |  107373 |                  5 |
| 2000 meatloaf                        |      475785 |        90 |          2202916 | 2012-03-06 00:00:00 | ['time-to-make', 'course', 'main-ingredient', 'preparation', 'main-dish', 'potatoes', 'vegetables', '4-hours-or-less', 'meatloaf', 'simply-potatoes2']                                                                                                                                             |        17 | ['pan fry bacon', 'and set aside on a paper towel to absorb excess grease', 'mince yellow onion', 'red bell pepper', 'and add to your mixing bowl', 'chop garlic and set aside', 'put 1tbsp olive oil into a saut pan', 'along with chopped garlic', 'teaspoons white pepper and a pinch of kosher salt', 'bring to a medium heat to sweat your garlic', 'preheat oven to 350f', 'coarsely chop your baby spinach add to your heated pan', 'stir frequently for approximately 5 min to wilt', 'add your spinach to the mixing bowl', 'chop your now cooled bacon', 'and add it to the mixing bowl', 'add your meatloaf mix to the bowl', 'with one egg and mix till thoroughly combined', 'add your goat cheese', 'one egg', '1 / 8 tsp white pepper and 1 / 8 tsp of kosher salt and mix till thoroughly combined', 'transfer to a 9x5 meatloaf pan', 'and cook for 60 min or until the internal temperature is at least 160f', 'let stand for 5min', 'melt 1tbsp unsalted butter into a frying pan', 'and cook up to three eggs at a time', 'crack each egg into a separate dish', 'in order to prevent egg shells from reaching the pan', 'then add salt and pepper to taste', 'wait until the egg whites are firm looking', 'but slightly runny on top before flipping your eggs', 'after flipping', 'wait 10~20 seconds before removing each egg and placing it over your slices of meatloaf'] | ready, set, cook! special edition contest entry: a mediterranean flavor inspired meatloaf dish. featuring: simply potatoes - shredded hash browns, egg, bacon, spinach, red bell pepper, and goat cheese.                                                                                                                                                                         | ['meatloaf mixture', 'unsmoked bacon', 'goat cheese', 'unsalted butter', 'eggs', 'baby spinach', 'yellow onion', 'red bell pepper', 'simply potatoes shredded hash browns', 'fresh garlic', 'kosher salt', 'white pepper', 'olive oil'] |              13 |      267   |             30 |         12 |          12 |           29 |                 48 |          2 |                  5 | False                 |             2012 | False                   |  176107 |                  5 |


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

```py
# Prepare data
rating_counts = interactions_df['rating'].value_counts().sort_index().reset_index()
rating_counts.columns = ['Rating', 'Count']

# Plot using Plotly Express
fig = px.bar(
    rating_counts,
    x='Rating',
    y='Count',
    title='Distribution of Recipe Ratings'
)
fig.show(renderer='jupyterlab')
```

<iframe src="assets/graph-1.html" width="800" height="600" frameborder="0"></iframe>

**Observation:**
This bar chart shows the frequency of each rating value (e.g., from 0 to 5).
We can observe how users rate recipes on average.

```py
# Filter recipes with preparation time <= 300 minutes
filtered_minutes = recipes_df[recipes_df['minutes'] <= 300]

# Plot using Plotly Express
fig = px.histogram(
    filtered_minutes,
    x='minutes',
    nbins=50,
    title='Distribution of Recipe Preparation Time (<= 300 minutes)',
    labels={'minutes': 'Minutes', 'count': 'Frequency'}
)
fig.show(renderer='jupyterlab')
```

<iframe src="assets/graph-2.html" width="800" height="600" frameborder="0"></iframe>

**Observation:**
The histogram displays how recipe preparation times are distributed.
We can see whether most recipes are quick to prepare or take longer.

```py
# Plot the distribution of number of steps
fig = recipes_df['n_steps'].plot(
    kind='hist',
    bins=30,
    title='Distribution of Number of Steps in Recipes',
    labels={'value': 'Number of Steps', 'count': 'Frequency'}
)
fig.show(renderer='jupyterlab')
```

<iframe src="assets/graph-3.html" width="800" height="600" frameborder="0"></iframe>

**Observation:**
This histogram shows the complexity of recipes based on the number of steps.
It helps us understand whether recipes tend to be simple or complex.

### Step: Bivariate Analysis

Examine relationships between pairs of variables to identify possible associations.

```py
# Filter out recipes with preparation time > 300 minutes for clarity
filtered_data = recipes_df[recipes_df['minutes'] <= 300]

# Create scatter plot
fig = filtered_data.plot(
    kind='scatter',
    x='minutes',
    y='average_rating',
    title='Preparation Time vs. Average Rating',
    labels={'minutes': 'Minutes', 'average_rating': 'Average Rating'}
)
fig.show(renderer='jupyterlab')
```

<iframe src="assets/graph-4.html" width="800" height="600" frameborder="0"></iframe>

**Observation:**
This scatter plot helps identify if there's a correlation between preparation time and average rating.
We can look for trends, such as whether quicker recipes tend to have higher ratings.

```py
# Create scatter plot
fig = recipes_df.plot(
    kind='scatter',
    x='n_steps',
    y='average_rating',
    title='Number of Steps vs. Average Rating',
    labels={'n_steps': 'Number of Steps', 'average_rating': 'Average Rating'}
)
fig.show(renderer='jupyterlab')
```

<iframe src="assets/graph-5.html" width="800" height="600" frameborder="0"></iframe>

**Observation:**
This plot examines if the complexity of a recipe (as measured by the number of steps) affects its average rating.
We can see if simpler recipes are rated higher.

### Step: Interesting Aggregates

We'll explore aggregate statistics by grouping and pivoting data.

```py
# Explode the 'tags' column so each tag is in a separate row
recipes_tags = recipes_df.explode('tags')
```

```py
# Group by tags and calculate the average rating
tag_ratings = recipes_tags.groupby('tags')['average_rating'].mean().reset_index()

# Remove tags with missing average ratings
tag_ratings = tag_ratings.dropna(subset=['average_rating'])

# Sort tags by average rating
top_tags = tag_ratings.sort_values(by='average_rating', ascending=False)
```

```py
# Display the top 10 tags
top_10_tags = top_tags.head(10)
```

```py
fig = top_10_tags.plot(
    kind='bar',
    x='tags',
    y='average_rating',
    title='Top 10 Tags by Average Rating',
    labels={'tags': 'Tag', 'average_rating': 'Average Rating'}
)
fig.update_layout(xaxis_tickangle=-45)
fig.show(renderer='jupyterlab')
```

<iframe src="assets/graph-6.html" width="800" height="600" frameborder="0"></iframe>

**Observation:**
We can see which tags are associated with higher-rated recipes. This helps identify popular cuisines or recipe categories.

```py
# Group by 'n_steps' and calculate average calories
steps_calories = recipes_df.groupby('n_steps')['calories'].mean().reset_index()
```

```py
# Plot the relationship
fig = steps_calories.plot(
    kind='scatter',
    x='n_steps',
    y='calories',
    title='Average Calories by Number of Steps',
    labels={'n_steps': 'Number of Steps', 'calories': 'Average Calories'}
)
fig.show(renderer='jupyterlab')
```

<iframe src="assets/graph-7.html" width="800" height="600" frameborder="0"></iframe>

**Observation:**
This plot shows how the average caloric content of recipes varies with the number of steps.
It can indicate whether more complex recipes tend to be higher or lower in calories.

---

## Step 3: Assessment of Missingness

### Step: NMAR Analysis

In this section, we'll explore whether any missing data in our dataset is **Not Missing At Random (NMAR)**. Recall that NMAR occurs when the probability of missingness is related to the missing values themselves, and not solely to observed data.

The `description` column in the `recipes_df` dataset has a number of missing values.

```py
# Check for missing values in 'description'
missing_descriptions = recipes_df['description'].isnull().sum()
total_recipes = recipes_df.shape[0]
print(f"Missing descriptions: {missing_descriptions} out of {total_recipes} recipes")
```

Missing descriptions: 70 out of 83782 recipes

We need to consider whether the missingness in the description column depends on the missing values themselves.

**Hypothesis:**

• Scenario 1 (NMAR): Recipe authors might omit the description because they believe the description is not necessary due to the simplicity or obviousness of the recipe. Alternatively, they might intentionally leave it blank if they have nothing special to mention about the recipe. In this case, the missingness depends on the content that would have been in the description (i.e., the description itself).

• Scenario 2 (Not NMAR): The missingness could be due to other factors, such as the experience level of the contributor, the time when the recipe was submitted, or whether the recipe is a variation of a common dish.

**Conclusion:**

• Without additional data or domain knowledge, it's plausible that the missingness in `description` is NMAR because the absence of a description might be related to the content that the contributor chose not to provide.

• For example, if a recipe is extremely simple (e.g., boiling eggs), the contributor might skip the description, thinking it's unnecessary. In this case, the missingness depends on the nature of the recipe itself, which is unobserved in the `description` field.

```py
# Check for missing values in 'review'
missing_reviews = interactions_df['review'].isnull().sum()
total_interactions = interactions_df.shape[0]
print(f"Missing reviews: {missing_reviews} out of {total_interactions} interactions")
```

Missing reviews: 169 out of 731927 interactions

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

```py
# Create a missingness indicator for 'description'
recipes_df['description_missing'] = recipes_df['description'].isnull()
```

```py
# Split the data into two groups based on 'description_missing'
has_description = recipes_df[recipes_df['description_missing'] == False]
missing_description = recipes_df[recipes_df['description_missing'] == True]

# Compute the mean 'n_steps' for each group
mean_n_steps_has_desc = has_description['n_steps'].mean()
mean_n_steps_missing_desc = missing_description['n_steps'].mean()

# Compute the observed difference in means
observed_diff = mean_n_steps_has_desc - mean_n_steps_missing_desc
print(f"Observed difference in mean n_steps: {observed_diff}")
```

Observed difference in mean n_steps: -0.9953913417431188

```py
def permutation_test(data, column, group_column, num_simulations=1000):
    observed = data.groupby(group_column)[column].mean()
    observed_diff = observed[False] - observed[True]  # False: has description, True: missing description

    diffs = []
    for _ in range(num_simulations):
        shuffled = data[column].sample(frac=1, replace=False).reset_index(drop=True)
        shuffled_data = data.copy()
        shuffled_data[column] = shuffled
        shuffled_means = shuffled_data.groupby(group_column)[column].mean()
        diff = shuffled_means[False] - shuffled_means[True]
        diffs.append(diff)

    p_value = np.mean(np.abs(diffs) >= np.abs(observed_diff))
    return observed_diff, diffs, p_value
```

```py
# Perform the permutation test
observed_diff, diffs, p_value = permutation_test(
    data=recipes_df,
    column='n_steps',
    group_column='description_missing',
    num_simulations=1000
)

print(f"P-value: {p_value}")
```

P-value: 0.208

```py
# Plot the distribution of permuted differences
fig = px.histogram(
    diffs,
    nbins=50,
    title='Permutation Test: Difference in Mean n_steps by Description Missingness',
    labels={'value': 'Difference in Mean n_steps'}
)

# Add a vertical line for the observed difference
fig.add_vline(x=observed_diff, line_color='red', line_dash='dash', annotation_text='Observed Difference')

# Show the plot
fig.show(renderer='jupyterlab')
```

<iframe src="assets/graph-8.html" width="800" height="600" frameborder="0"></iframe>

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

```py
# Create an indicator for missing 'description'
recipes_df['description_missing'] = recipes_df['description'].isna()
```

```py
# Ensure 'submitted' is in datetime format
recipes_df['submitted'] = pd.to_datetime(recipes_df['submitted'])

# Extract the year from 'submitted' date
recipes_df['submitted_year'] = recipes_df['submitted'].dt.year
```

**Visualize Missingness Over Time:**

```py
# Assuming recipes_df is already loaded and 'submitted' is in datetime format

# Create an indicator for missing 'description'
recipes_df['description_missing'] = recipes_df['description'].isna()

# Extract the year from 'submitted' date
recipes_df['submitted_year'] = recipes_df['submitted'].dt.year

# Calculate the proportion of missing descriptions per year
missing_by_year = recipes_df.groupby('submitted_year')['description_missing'].mean()

# Reset the index to make 'submitted_year' a column
missing_by_year = missing_by_year.reset_index()

# Plot using the .plot() method
fig = missing_by_year.plot(
    kind='bar',
    x='submitted_year',
    y='description_missing',
    title='Proportion of Missing Descriptions by Submission Year',
    labels={'submitted_year': 'Submission Year', 'description_missing': 'Proportion Missing'}
)

# Show the plot
fig.show(renderer='jupyterlab')
```

<iframe src="assets/graph-9.html" width="800" height="600" frameborder="0"></iframe>

**Interpretation:**

• If the proportion of missing descriptions varies significantly over the years, it suggests that missingness depends on the submission year.

**Defining Groups Based on Submission Year:**

To perform a statistical test, we'll divide the data into two groups:

• Group A: Recipes submitted before a certain year (e.g., 2010).

• Group B: Recipes submitted in or after that year.

```py
# Choose a cutoff year (e.g., 2010)
cutoff_year = 2010

# Create a binary variable
recipes_df['submitted_before_2010'] = recipes_df['submitted_year'] < cutoff_year
```

**Calculating the Observed Difference:**

```py
# Group by 'submitted_before_2010' and calculate the mean of 'description_missing'
grouped = recipes_df.groupby('submitted_before_2010')['description_missing'].mean()

# Extract proportions
prop_before = grouped[True]
prop_after = grouped[False]

# Calculate the observed difference
observed_diff = prop_before - prop_after

print(f"Proportion missing (Submitted before 2010): {prop_before:.4f}")
print(f"Proportion missing (Submitted in/after 2010): {prop_after:.4f}")
print(f"Observed difference in proportions: {observed_diff:.4f}")
```

Proportion missing (Submitted before 2010): 0.0011
Proportion missing (Submitted in/after 2010): 0.0004
Observed difference in proportions: 0.0007

**Performing the Permutation Test:**

```py
# Number of permutations
num_permutations = 1000

# Array to store permutation differences
perm_diffs = np.zeros(num_permutations)

# Extract arrays
description_missing = recipes_df['description_missing'].values
submitted_before_2010 = recipes_df['submitted_before_2010'].values
```

```py
for i in range(num_permutations):
    # Shuffle the 'submitted_before_2010' labels
    shuffled_labels = np.random.permutation(submitted_before_2010)
    
    # Create a DataFrame with shuffled labels
    shuffled_df = pd.DataFrame({
        'description_missing': description_missing,
        'shuffled_submitted_before_2010': shuffled_labels
    })
    
    # Calculate the proportion of missing descriptions for each group
    shuffled_grouped = shuffled_df.groupby('shuffled_submitted_before_2010')['description_missing'].mean()
    
    # Calculate the difference in proportions
    perm_diff = shuffled_grouped[True] - shuffled_grouped[False]
    
    # Store the permutation difference
    perm_diffs[i] = perm_diff
```

```py
# Calculate the p-value
p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

print(f"P-value: {p_value:.4f}")
```

P-value: 0.0000

```py
# Plot the distribution of permuted differences
fig = px.histogram(
    perm_diffs,
    nbins=50,
    title='Permutation Test: Difference in Proportion of Missing Descriptions by Submission Date',
    labels={'value': 'Difference in Proportions'}
)

# Add a vertical line for the observed difference
fig.add_vline(
    x=observed_diff,
    line_color='red',
    line_dash='dash',
    annotation_text='Observed Difference',
    annotation_position='top left'
)

# Show the plot
fig.show(renderer='jupyterlab')
```

<iframe src="assets/graph-10.html" width="800" height="600" frameborder="0"></iframe>

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

```py
# Merge datasets on 'recipe_id'
merged_df = pd.merge(interactions_df, recipes_df, on='recipe_id', how='inner')
```

```py
# Select necessary columns and drop rows with missing values
analysis_df = merged_df[['recipe_id', 'minutes', 'rating']].dropna()
```

```py
# Categorize recipes based on preparation time
analysis_df['time_category'] = np.where(analysis_df['minutes'] <= 30, 'short', 'long')
```

### Step: Calculate Observed Test Statistic

Compute the mean ratings for each group and calculate the observed difference.

```py
# Calculate mean ratings for each group
group_means = analysis_df.groupby('time_category')['rating'].mean()
mean_short = group_means['short']
mean_long = group_means['long']

# Calculate observed test statistic
observed_diff = mean_short - mean_long
print(f"Observed Difference in Mean Ratings: {observed_diff}")
```

Observed Difference in Mean Ratings: 0.12769331475557077

#### Permutation Test:

Perform a permutation test to assess whether the observed difference is statistically significant.

**Steps:**

• Combine All Ratings: Under the null hypothesis, the grouping is arbitrary.

• Shuffle Time Categories: Randomly assign 'short' and 'long' labels to the ratings.

• Recalculate the Difference in Means: For each permutation, compute the difference in mean ratings.

• Repeat: Perform this process multiple times to build a distribution under the null hypothesis.

• Calculate P-value: Determine the proportion of permutations where the absolute permuted difference is greater than or equal to the observed difference.

**Implementation:**

```py
# Number of permutations
num_permutations = 1000

# Store permuted differences
perm_diffs = []

# Perform permutations
for _ in range(num_permutations):
    # Shuffle the 'time_category' labels
    shuffled_labels = analysis_df['time_category'].sample(frac=1, replace=False).reset_index(drop=True)
    
    # Assign shuffled labels
    shuffled_df = analysis_df.copy()
    shuffled_df['shuffled_category'] = shuffled_labels
    
    # Calculate mean ratings for shuffled groups
    shuffled_means = shuffled_df.groupby('shuffled_category')['rating'].mean()
    perm_diff = shuffled_means['short'] - shuffled_means['long']
    perm_diffs.append(perm_diff)

# Convert list to numpy array
perm_diffs = np.array(perm_diffs)

# Calculate p-value
p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
print(f"P-value: {p_value}")
```

P-value: 0.0

**Visualization:** Plot the distribution of permuted differences and indicate the observed difference.

```py
# Create a histogram of permuted differences
fig = px.histogram(
    perm_diffs,
    nbins=50,
    title='Permutation Test: Distribution of Difference in Mean Ratings',
    labels={'value': 'Difference in Mean Ratings'}
)

# Add a vertical line for the observed difference
fig.add_vline(
    x=observed_diff,
    line_color='red',
    line_dash='dash',
    annotation_text='Observed Difference',
    annotation_position='top left'
)

# Show the plot
fig.show(renderer='jupyterlab')
```

<iframe src="assets/graph-11.html" width="800" height="600" frameborder="0"></iframe>

**Conclusion:**

```py
if p_value < 0.05:
    print("Based on the p-value calculated from the permutation test: We reject the null hypothesis at the 5% significance level.")
else:
    print("Based on the p-value calculated from the permutation test: We fail to reject the null hypothesis at the 5% significance level.")
```

Based on the p-value calculated from the permutation test: We reject the null hypothesis at the 5% significance level.

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

```py
# Import scikit-learn libraries
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

### Step: Data Preparation

Assuming that the datasets `recipes_df` and `interactions_df` have been loaded and preprocessed as per previous steps.

```py
# Load the datasets (if not already loaded)
# recipes_df = pd.read_csv('RAW_recipes.csv')
# interactions_df = pd.read_csv('RAW_interactions.csv')

# Ensure 'submitted' and 'date' columns are datetime
recipes_df['submitted'] = pd.to_datetime(recipes_df['submitted'])
interactions_df['date'] = pd.to_datetime(interactions_df['date'])

# Merge datasets on 'recipe_id' to include average ratings
# Calculate the average rating per recipe
average_ratings = interactions_df.groupby('recipe_id')['rating'].mean().reset_index()
average_ratings = average_ratings.rename(columns={'rating': 'average_rating'})

# Merge average ratings with recipes_df
recipes_with_ratings = pd.merge(recipes_df, average_ratings, left_on='recipe_id', right_on='recipe_id', how='inner')

# Display the first few rows
recipes_with_ratings.head()
```

| name                                 |   recipe_id |   minutes |   contributor_id | submitted           | tags                                                                                                                                                                                                                                                                                               |   n_steps | steps                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | description                                                                                                                                                                                                                                                                                                                                                                       | ingredients                                                                                                                                                                                                                             |   n_ingredients |   calories |   total_fat_DV |   sugar_DV |   sodium_DV |   protein_DV |   saturated_fat_DV |   carbs_DV |   average_rating_x | description_missing   |   submitted_year | submitted_before_2010   |   average_rating_y |
|:-------------------------------------|------------:|----------:|-----------------:|:--------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|-----------:|---------------:|-----------:|------------:|-------------:|-------------------:|-----------:|-------------------:|:----------------------|-----------------:|:------------------------|-------------------:|
| 1 brownies in the world    best ever |      333281 |        40 |           985201 | 2008-10-27 00:00:00 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings']                                                                        |        10 | ['heat the oven to 350f and arrange the rack in the middle', 'line an 8-by-8-inch glass baking dish with aluminum foil', 'combine chocolate and butter in a medium saucepan and cook over medium-low heat', 'stirring frequently', 'until evenly melted', 'remove from heat and let cool to room temperature', 'combine eggs', 'sugar', 'cocoa powder', 'vanilla extract', 'espresso', 'and salt in a large bowl and briefly stir until just evenly incorporated', 'add cooled chocolate and mix until uniform in color', 'add flour and stir until just incorporated', 'transfer batter to the prepared baking dish', 'bake until a tester inserted in the center of the brownies comes out clean', 'about 25 to 30 minutes', 'remove from the oven and cool completely before cutting']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | these are the most; chocolatey, moist, rich, dense, fudgy, delicious brownies that you'll ever make.....sereiously! there's no doubt that these will be your fav brownies ever for you can add things to them or make them plain.....either way they're pure heaven!                                                                                                              | ['bittersweet chocolate', 'unsalted butter', 'eggs', 'granulated sugar', 'unsweetened cocoa powder', 'vanilla extract', 'brewed espresso', 'kosher salt', 'all-purpose flour']                                                          |               9 |      138.4 |             10 |         50 |           3 |            3 |                 19 |          6 |                  4 | False                 |             2008 | True                    |                  4 |
| 1 in canada chocolate chip cookies   |      453467 |        45 |          1848091 | 2011-04-11 00:00:00 | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings']                                                                                                                                      |        12 | ['pre-heat oven the 350 degrees f', 'in a mixing bowl', 'sift together the flours and baking powder', 'set aside', 'in another mixing bowl', 'blend together the sugars', 'margarine', 'and salt until light and fluffy', 'add the eggs', 'water', 'and vanilla to the margarine / sugar mixture and mix together until well combined', 'add in the flour mixture to the wet ingredients and blend until combined', 'scrape down the sides of the bowl and add the chocolate chips', 'mix until combined', 'scrape down the sides to the bowl again', 'using an ice cream scoop', 'scoop evenly rounded balls of dough and place of cookie sheet about 1 - 2 inches apart to allow for spreading during baking', 'bake for 10 - 15 minutes or until golden brown on the outside and soft & chewy in the center', 'serve hot and enjoy !']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | this is the recipe that we use at my school cafeteria for chocolate chip cookies. they must be the best chocolate chip cookies i have ever had! if you don't have margarine or don't like it, then just use butter (softened) instead.                                                                                                                                            | ['white sugar', 'brown sugar', 'salt', 'margarine', 'eggs', 'vanilla', 'water', 'all-purpose flour', 'whole wheat flour', 'baking soda', 'chocolate chips']                                                                             |              11 |      595.1 |             46 |        211 |          22 |           13 |                 51 |         26 |                  5 | False                 |             2011 | False                   |                  5 |
| 412 broccoli casserole               |      306168 |        40 |            50969 | 2008-05-30 00:00:00 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                                                                                               |         6 | ['preheat oven to 350 degrees', 'spray a 2 quart baking dish with cooking spray', 'set aside', 'in a large bowl mix together broccoli', 'soup', 'one cup of cheese', 'garlic powder', 'pepper', 'salt', 'milk', '1 cup of french onions', 'and soy sauce', 'pour into baking dish', 'sprinkle remaining cheese over top', 'bake for 25 minutes or until cheese is lightly browned', 'sprinkle with rest of french fried onions and bake until onions are browned and cheese is bubbly', 'about 10 more minutes']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one  #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions']                                                                   |               9 |      194.8 |             20 |          6 |          32 |           22 |                 36 |          3 |                  5 | False                 |             2008 | True                    |                  5 |
| millionaire pound cake               |      286009 |       120 |           461724 | 2008-02-12 00:00:00 | ['time-to-make', 'course', 'cuisine', 'preparation', 'occasion', 'north-american', 'desserts', 'american', 'southern-united-states', 'dinner-party', 'holiday-event', 'cakes', 'dietary', 'christmas', 'thanksgiving', 'low-sodium', 'low-in-something', 'taste-mood', 'sweet', '4-hours-or-less'] |         7 | ['freheat the oven to 300 degrees', 'grease a 10-inch tube pan with butter', 'dust the bottom and sides with flour', 'and set aside', 'in a large mixing bowl', 'cream the butter and sugar with an electric mixer and add the eggs one at a time', 'beating after each addition', 'alternately add the flour and milk', 'stirring till the batter is smooth', 'add the two extracts and stir till well blended', 'scrape the batter into the prepared pan and bake till a cake tester or knife blade inserted in the center comes out clean', 'about 1 1 / 2 hours', 'cool the cake in the pan on a rack for 5 minutes', 'then turn it out on the rack to cool completely']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | why a millionaire pound cake?  because it's super rich!  this scrumptious cake is the pride of an elderly belle from jackson, mississippi.  the recipe comes from "the glory of southern cooking" by james villas.                                                                                                                                                                | ['butter', 'sugar', 'eggs', 'all-purpose flour', 'whole milk', 'pure vanilla extract', 'almond extract']                                                                                                                                |               7 |      878.3 |             63 |        326 |          13 |           20 |                123 |         39 |                  5 | False                 |             2008 | True                    |                  5 |
| 2000 meatloaf                        |      475785 |        90 |          2202916 | 2012-03-06 00:00:00 | ['time-to-make', 'course', 'main-ingredient', 'preparation', 'main-dish', 'potatoes', 'vegetables', '4-hours-or-less', 'meatloaf', 'simply-potatoes2']                                                                                                                                             |        17 | ['pan fry bacon', 'and set aside on a paper towel to absorb excess grease', 'mince yellow onion', 'red bell pepper', 'and add to your mixing bowl', 'chop garlic and set aside', 'put 1tbsp olive oil into a saut pan', 'along with chopped garlic', 'teaspoons white pepper and a pinch of kosher salt', 'bring to a medium heat to sweat your garlic', 'preheat oven to 350f', 'coarsely chop your baby spinach add to your heated pan', 'stir frequently for approximately 5 min to wilt', 'add your spinach to the mixing bowl', 'chop your now cooled bacon', 'and add it to the mixing bowl', 'add your meatloaf mix to the bowl', 'with one egg and mix till thoroughly combined', 'add your goat cheese', 'one egg', '1 / 8 tsp white pepper and 1 / 8 tsp of kosher salt and mix till thoroughly combined', 'transfer to a 9x5 meatloaf pan', 'and cook for 60 min or until the internal temperature is at least 160f', 'let stand for 5min', 'melt 1tbsp unsalted butter into a frying pan', 'and cook up to three eggs at a time', 'crack each egg into a separate dish', 'in order to prevent egg shells from reaching the pan', 'then add salt and pepper to taste', 'wait until the egg whites are firm looking', 'but slightly runny on top before flipping your eggs', 'after flipping', 'wait 10~20 seconds before removing each egg and placing it over your slices of meatloaf'] | ready, set, cook! special edition contest entry: a mediterranean flavor inspired meatloaf dish. featuring: simply potatoes - shredded hash browns, egg, bacon, spinach, red bell pepper, and goat cheese.                                                                                                                                                                         | ['meatloaf mixture', 'unsmoked bacon', 'goat cheese', 'unsalted butter', 'eggs', 'baby spinach', 'yellow onion', 'red bell pepper', 'simply potatoes shredded hash browns', 'fresh garlic', 'kosher salt', 'white pepper', 'olive oil'] |              13 |      267   |             30 |         12 |          12 |           29 |                 48 |          2 |                  5 | False                 |             2012 | False                   |                  5 |

### Step: Feature Selection

We'll select at least two features to use in our baseline model:

• **Numerical Features:** `minutes`, `n_ingredients`, `n_steps`, `calories`, `protein_DV`, `carbs_DV`

Our target variable will be `average_rating`.

### Step: Prepare the Data

Prepare the features (`X`) and the target variable (`y`), and check for missing values.

```py
# Merge interactions with recipes to get features and ratings
data = pd.merge(interactions_df, recipes_df, on='recipe_id', how='inner')
```

```py
# Select additional features
X = data[['minutes', 'n_ingredients', 'n_steps', 'calories', 'protein_DV', 'carbs_DV']]
y = data['rating']

# Check for missing values in features
print("Missing values in features:")
print(X.isnull().sum())

# Check for missing values in target variable
print("\nMissing values in target variable:")
print(y.isnull().sum())
```

Missing values in features:
minutes          0
n_ingredients    0
n_steps          0
calories         0
protein_DV       0
carbs_DV         0
dtype: int64

Missing values in target variable:
0

```py
# First, ensure 'tags' is properly parsed into lists
# Explode the 'tags' column
data_exploded = data.explode('tags')

# Create dummy variables for tags
tags_dummies = pd.get_dummies(data_exploded['tags'])

# Aggregate dummy variables by 'recipe_id' and 'user_id'
tags_dummies_grouped = tags_dummies.groupby([data_exploded['recipe_id'], data_exploded['user_id']]).max().reset_index()

# Merge back with the main data
data = pd.merge(data, tags_dummies_grouped, on=['recipe_id', 'user_id'], how='left')
```

### Step: Split Data into Training and Testing Sets

Split the data into training and testing sets to evaluate the model's ability to generalize to unseen data.

```py
# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
```

Training set size: 187542 samples
Testing set size: 46886 samples

### Step: Create a Pipeline

Create a scikit-learn Pipeline that includes data preprocessing (scaling) and model training.

```py
# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),         # Scale numerical features
    ('regressor', LinearRegression())     # Linear Regression model
])
```

### Step: Train the Baseline Model

```py
# Fit the model on the training data
pipeline.fit(X_train, y_train)
```

### Step: Evaluate the Model

Evaluate the model's performance on the test set.

```py
# Predict on the test set
y_pred = pipeline.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared Score: {r2:.4f}")
```

Mean Squared Error (MSE): 1.7862
Mean Absolute Error (MAE): 0.8978
R-squared Score: 0.0025

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

```py
# Merge interactions with recipes to get features and ratings
data = pd.merge(interactions_df, recipes_df, on='recipe_id', how='inner')
```

```py
# Drop rows with missing values in relevant columns
data = data.dropna(subset=['minutes', 'n_ingredients', 'n_steps', 'tags', 'rating'])
```

```py
# Calculate minutes_per_step
data['minutes_per_step'] = data['minutes'] / data['n_steps']

# Calculate ingredients_per_step
data['ingredients_per_step'] = data['n_ingredients'] / data['n_steps']
```

#### Handle Infinite or NaN Values:

• Recipes with `n_steps` equal to zero will result in division by zero.

• Replace infinite values with zero or appropriate value.

```py
# Replace infinite values with NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values resulting from division
data = data.dropna(subset=['minutes_per_step', 'ingredients_per_step'])
```

### Encoding Categorical Variables

Assuming 'tags' is already a list of tags per recipe

#### Select Most Frequent Tags

• Select the top 20 most frequent tags to reduce dimensionality.

```py
# Explode the 'tags' column
data_exploded = data.explode('tags')

# Get the most frequent tags
top_tags = data_exploded['tags'].value_counts().head(20).index.tolist()
```

#### Create Dummy Variables for Top Tags

```py
# Create a binary feature for each top tag
for tag in top_tags:
    data[f'tag_{tag}'] = data['tags'].apply(lambda x: int(tag in x))
```

#### Feature Selection

**Our final set of features (X) includes:**

• **Numerical Features:** `minutes`, `n_ingredients`, `n_steps`, `minutes_per_step`, `ingredients_per_step`

• **Categorical Features (Encoded):** Binary features for each of the top 20 tags

The target variable (`y`) is the individual `rating`.

### Split Data into Training and Testing Sets

We will use the same train-test split as in the baseline model for consistency.

```py
# Select features and target variable
feature_columns = ['minutes', 'n_ingredients', 'n_steps', 'minutes_per_step', 'ingredients_per_step'] + [f'tag_{tag}' for tag in top_tags]
X = data[feature_columns]
y = data['rating']

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
```

Training set size: 187542 samples
Testing set size: 46886 samples

#### Using RandomizedSearchCV for Hyperparameter Tuning

**RandomizedSearchCV** is a method provided by scikit-learn that allows you to perform hyperparameter tuning by sampling a fixed number of parameter settings from specified distributions. Instead of exhaustively trying all possible combinations (as in Grid Search), Randomized Search evaluates a random selection of combinations, which can significantly reduce computation time while still exploring a wide range of hyperparameters.

```py
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=42))
])
```

#### Define the Hyperparameter Distributions

Define distributions from which parameter values will be sampled.

```py
# Define hyperparameter distributions
param_distributions = {
    # 'regressor__n_estimators': randint(50, 200),
    'regressor__n_estimators': [50, 100],  # Fixed options instead of a range
    'regressor__max_depth': [None, 10],
    'regressor__min_samples_leaf': [1, 2]
}
```

#### Set Up RandomizedSearchCV

**Parameters:**

• `n_iter`: The number of parameter settings that are sampled. You can adjust this number based on available resources.

• `random_state`: Ensures reproducibility of results.

```py
# Randomized search
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=5,
    cv=3,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    random_state=42,
    verbose=2
)

# Sample 20% of the training data
X_train_sample = X_train.sample(frac=0.2, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]

# Use the sampled data for hyperparameter tuning
random_search.fit(X_train_sample, y_train_sample)
```

Fitting 3 folds for each of 5 candidates, totalling 15 fits

```py
# Best hyperparameters
print("Best Hyperparameters:")
print(random_search.best_params_)
```

Best Hyperparameters:
{'regressor__n_estimators': 100, 'regressor__min_samples_leaf': 2, 'regressor__max_depth': 10}

```py
# Evaluate on test set
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Final Model Performance with RandomizedSearchCV:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared Score: {r2:.4f}")
```

Final Model Performance with RandomizedSearchCV:
Mean Squared Error (MSE): 1.7737
Mean Absolute Error (MAE): 0.8911
R-squared Score: 0.0095

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

```py
# Ensure that the 'tag_easy' feature is included in the test set
if 'tag_easy' not in X_test.columns:
    X_test['tag_easy'] = X_test['tags'].apply(lambda x: int('easy' in x))
```

#### Identify the Two Groups

```py
# Create a boolean mask for the two groups
easy_mask = X_test['tag_easy'] == 1
not_easy_mask = X_test['tag_easy'] == 0

# Separate the test data into the two groups
X_test_easy = X_test[easy_mask]
y_test_easy = y_test[easy_mask]
X_test_not_easy = X_test[not_easy_mask]
y_test_not_easy = y_test[not_easy_mask]
```

#### Make Predictions for Each Group

```py
# Predict for "easy" recipes
y_pred_easy = best_model.predict(X_test_easy)

# Predict for "not easy" recipes
y_pred_not_easy = best_model.predict(X_test_not_easy)
```

#### Calculate the Observed RMSE for Each Group

```py
from sklearn.metrics import mean_squared_error
import numpy as np

# Calculate RMSE for each group
rmse_easy = np.sqrt(mean_squared_error(y_test_easy, y_pred_easy))
rmse_not_easy = np.sqrt(mean_squared_error(y_test_not_easy, y_pred_not_easy))

# Calculate the observed difference in RMSE
observed_diff_rmse = rmse_easy - rmse_not_easy
print(f"Observed RMSE (Easy Recipes): {rmse_easy:.4f}")
print(f"Observed RMSE (Not Easy Recipes): {rmse_not_easy:.4f}")
print(f"Observed Difference in RMSE: {observed_diff_rmse:.4f}")
```

Observed RMSE (Easy Recipes): 1.2961
Observed RMSE (Not Easy Recipes): 1.3791
Observed Difference in RMSE: -0.0831

### Permutation Test

#### Combine Residuals and Group Labels

First, we need to compute the residuals (errors) for all test samples.

```py
# Combine all test predictions and true values
y_test_all = pd.concat([y_test_easy, y_test_not_easy])
y_pred_all = np.concatenate([y_pred_easy, y_pred_not_easy])

# Calculate residuals
residuals = y_test_all - y_pred_all

# Create a DataFrame with residuals and group labels
residuals_df = pd.DataFrame({
    'residuals': residuals,
    'group': ['easy'] * len(y_test_easy) + ['not_easy'] * len(y_test_not_easy)
})
```

#### Permutation Procedure

```py
# Number of permutations
num_permutations = 1000

# Store permuted differences
perm_diffs = []

# Observed group sizes
n_easy = len(y_test_easy)
n_not_easy = len(y_test_not_easy)

for _ in range(num_permutations):
    # Shuffle the group labels
    shuffled_labels = residuals_df['group'].sample(frac=1, replace=False).reset_index(drop=True)
    
    # Assign shuffled labels
    residuals_df['shuffled_group'] = shuffled_labels
    
    # Calculate RMSE for each permuted group
    rmse_easy_perm = np.sqrt(np.mean(residuals_df[residuals_df['shuffled_group'] == 'easy']['residuals'] ** 2))
    rmse_not_easy_perm = np.sqrt(np.mean(residuals_df[residuals_df['shuffled_group'] == 'not_easy']['residuals'] ** 2))
    
    # Calculate the difference in RMSE
    perm_diff = rmse_easy_perm - rmse_not_easy_perm
    perm_diffs.append(perm_diff)
```

#### Calculate the P-value

```py
# Convert permuted differences to a numpy array
perm_diffs = np.array(perm_diffs)

# Calculate the p-value
p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff_rmse))
print(f"P-value: {p_value:.4f}")
```

P-value: 0.0200

#### Visualization

```py
import plotly.express as px

# Create a histogram of permuted differences
fig = px.histogram(
    perm_diffs,
    nbins=50,
    title='Permutation Test: Distribution of Difference in RMSE',
    labels={'value': 'Difference in RMSE'}
)

# Add a vertical line for the observed difference
fig.add_vline(
    x=observed_diff_rmse,
    line_color='red',
    line_dash='dash',
    annotation_text='Observed Difference',
    annotation_position='top left'
)

# Show the plot
fig.show(renderer='jupyterlab')
```

<iframe src="assets/graph-12.html" width="800" height="600" frameborder="0"></iframe>

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