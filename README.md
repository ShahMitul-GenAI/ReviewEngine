# Customer Review Engine

This project automates the process of harvesting and summarizing Amazon customer reviews for a specified product. Using OpenAI, it provides a concise summary of customer feedback, helping users quickly understand the general sentiment and key points from multiple reviews.

## Features

- **Product Description Input:** Users can describe the product using 2-5 common phrases.
- **Review Count Specification:** Users can specify the number of customer reviews to pull for summary generation.
- **Top Customer Reviews:** The output includes the top 10 high ranking customer reviews.
- **Overall Review Summary:** A summarized overview of the collected reviews, generated using OpenAI.

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shahmitul-genai/ReviewEngine.git
   cd amazon-review-summarizer
   ```

2. Navigate to the project directory
    ```bash
    cd src/cust_reviews
    ```

3. Install Poetry using pip (if not already installed):
   ```bash
   pip install poetry
   ```

4. Activate the virtual environment created by Poetry:
    ``` bash
    poetry shell
    ```

5. Install project dependencies using Poetry:
    ```
    poetry install
    ```

### Configuration

1. Create a `.env` file and add your own OpenAI API key in the `.env` file as follows:
   ```bash
   OPENAI_API_KEY='your-key-here'
   ```

### Usage

1. After installing the dependencies, you can run the Streamlit app in root directory by executing the following command:
   ```bash
   streamlit run app_cr.py
   ```

2. Follow the prompts to input the product description 

3. Move the slide bar to select number of reviews to pull.

3. The script will display the top 10 ranking customer reviews and the overall review summary.

### Example

Here’s an example of how to use the program:

1. **Describe the product:** "wireless earbuds, noise cancelling, Bluetooth 5.0"
2. **Specify number of reviews:** 25

**Output:**

- **Top 10 Customer Reviews:**
  '''
  A dataframe showing customer name, review rating, detailed customer reviews for the top 10 high-rating customer. 
  ...
  

- **Overall Review Summary:**
  "The wireless earbuds are highly praised for their sound quality and noise cancellation. However, some customers find them slightly uncomfortable. The battery life is generally considered excellent, making these earbuds a great value for the price."

## Contributing

We welcome contributions to enhance the functionality and usability of this project. Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.