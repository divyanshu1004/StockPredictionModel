# Stock Market Predictor 📈

A deep learning-based stock price prediction application that uses LSTM neural networks to forecast stock prices. Built with TensorFlow/Keras and Streamlit for an interactive web interface.

## Features

- **Real-time Stock Data**: Fetches historical stock data using Yahoo Finance API
- **LSTM Model**: Uses a multi-layer LSTM neural network trained on historical price data
- **Technical Analysis**: Displays moving averages (MA50, MA100, MA200) for trend analysis
- **Price Predictions**: Predicts stock prices for the next 30 days
- **Interactive Dashboard**: Web-based interface built with Streamlit
- **Visual Analytics**: Multiple charts showing price trends, moving averages, and predictions
- **Flexible Data Range**: Automatically handles up to 10 years of historical data
- **Data Validation**: Ensures sufficient historical data (minimum 100 days) for accurate predictions

## Demo

The application displays:
- Historical stock data table
- Price vs MA50 chart
- Price vs MA50 vs MA100 chart
- Price vs MA100 vs MA200 chart
- Original vs Predicted prices with 30-day future forecast

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd StockPredictionModel
```

2. Install required dependencies:
```bash
pip install numpy pandas matplotlib yfinance keras tensorflow streamlit scikit-learn
```

3. Ensure the trained model file is present:
   - `Stock Preictions Model.keras` should be in the project root directory

## Usage

### Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL displayed (typically `http://localhost:8501`)

3. Enter a stock symbol in the input field:
   - For Indian stocks: Use format like `HDFCBANK.NS`, `TCS.NS`, `RELIANCE.NS`
   - For US stocks: Use format like `AAPL`, `GOOGL`, `MSFT`

4. View the analysis and predictions in the dashboard

### Training Your Own Model

To train a new model with custom data:

1. Open the Jupyter notebook:
```bash
jupyter notebook stock_price_predictor.ipynb
```

2. Modify the stock symbol and parameters as needed

3. Run all cells to:
   - Fetch historical data
   - Prepare training/test datasets
   - Build and train the LSTM model
   - Evaluate predictions
   - Save the trained model

## Model Architecture

The LSTM model consists of:
- **Input Layer**: 100 timesteps (100 days of historical prices)
- **LSTM Layer 1**: 50 units with ReLU activation and 20% dropout
- **LSTM Layer 2**: 60 units with ReLU activation and 30% dropout
- **LSTM Layer 3**: 80 units with ReLU activation and 40% dropout
- **LSTM Layer 4**: 120 units with ReLU activation and 50% dropout
- **Output Layer**: Dense layer with 1 unit (predicted price)

The model is trained using:
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error
- **Training Split**: 80% training, 20% testing
- **Data Scaling**: MinMaxScaler (0-1 normalization)

## Project Structure

```
StockPredictionModel/
├── app.py                          # Streamlit web application
├── stock_price_predictor.ipynb     # Jupyter notebook for model training
├── Stock Preictions Model.keras    # Pre-trained LSTM model
└── README.md                        # Project documentation
```

## How It Works

1. **Data Collection**: Fetches up to 10 years of historical stock data from Yahoo Finance
2. **Data Preprocessing**: 
   - Splits data into 80% training and 20% testing
   - Normalizes prices using MinMaxScaler
   - Creates sequences of 100 days for LSTM input
3. **Prediction**:
   - Uses the last 100 days to predict the next day's price
   - Recursively predicts 30 days into the future
4. **Visualization**: Displays results with matplotlib charts in Streamlit interface

## Dependencies

```
numpy
pandas
matplotlib
yfinance
keras
tensorflow
streamlit
scikit-learn
```

## Limitations

- Requires minimum 100 days of historical data for predictions
- Stock market predictions are inherently uncertain and should not be used as sole investment advice
- Model performance depends on data quality and market volatility
- Past performance does not guarantee future results

## Support

For issues, questions, or contributions:
- Open an issue in the repository
- Review the code documentation in `app.py` and `stock_price_predictor.ipynb`

## Disclaimer

This tool is for educational and research purposes only. Stock market predictions are inherently uncertain and involve significant risk. Always conduct thorough research and consult with financial advisors before making investment decisions. The developers are not responsible for any financial losses incurred from using this application.

## License

This project is available for educational purposes. Please check with the repository owner for licensing details.

## Acknowledgments

- **yfinance**: For providing easy access to Yahoo Finance data
- **TensorFlow/Keras**: For deep learning framework
- **Streamlit**: For the intuitive web interface framework
