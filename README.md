
# Friend Recommendation System

This repository contains a Flask-based API for recommending friends based on user data. It uses cosine similarity to find similar user profiles and suggests potential friends.

## Prerequisites

- Python 3.x
- Flask
- pandas
- numpy
- scikit-learn

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/sayantanmaji10/FriendConnect.git


2. Install the required dependencies:

    ```shell
   pip install -r requirements.txt


## Usage
1. Ensure that the user dataset is available as a JSON file named  'jessusers.json'. Make sure the dataset follows the expected format.

2. Start the Flask server by running the following command:

   ```shell
   python app.py

3. Once the server is running, we can access the API through the following endpoint:   

   ```bash
   http://localhost:80/predict?email=<user_email>&limit=<limit_value>&page=<page_number>

A. <user_email>: The email address of the user for whom friend recommendations are required.

B. <limit_value>: The maximum number of recommendations to return per page.

C. <page_number>: The page number of recommendations to retrieve.

4. The API will return a JSON response containing a list of recommended friends' profiles, including their first name, last name, email, profile image, and gender.


## Examples

Example API request:
   ```bash
   http://localhost:80/predict?email=user@example.com&limit=10&page=1

```
##  Example API response:
```json
[
  {
    "firstname": "Steven",
    "lastname": "Smith",
    "friend_email": "steven.smith@example.com",
    "profileImage": "https://example.com/profiles/stevensmith.jpg",
    "gender": "Male"
  },

  {
    "firstname": "Elyse",
    "lastname": "Perry",
    "friend_email": "elyse.perry@example.com",
    "profileImage": "https://example.com/profiles/elyseperry.jpg",
    "gender": "Female"
  },
  
]
```









