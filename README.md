# ECON 310 Stock Market Predictions
Machine Learning model to pick stocks for the Stock Market game in ECON 310 at University of Michigan

## Game Criteria
Pick a stock on Monday 9:30am that will outperform the SP500 for the coming week, Monday Open through Friday Close. The change is measured by the percent change calculated as follows:
$$\%\Delta=\frac{\$Open_{monday}-\$Close_{friday}}{\$Open_{monday}}\cdot 100\%$$

**Win criteria**: If the $\%\Delta$ of the selected stock exceeds that of the SP500, as measured by the SPX index, you have won on a given week.

## Data
All data comes from yfinance and the model was trained each week from data on 2020-01-01 to the Friday prior to the Monday where predictions are entered.

## Data Preprocessing
Data cleaning, feature engineering, and data preprocessing was done prior to model training.

### Feature Engineering and Target Creation
**Engineered Features**
- Daily Return
- SP500 Daily Return
- Daily Outperformed SP500
- Rolling 5-day Outpreform
- 5-day MA
- 10-day MA
- 5-day Momentum
- 5-day Volumne Change
- 5-day Volatility
- Price-to-Volume
- MACD
- 12-Day EMA
- 26-day EMA
- Signal Line
- ROC
- 52-week High
- % Distance from 52-week High
- 52-week Low
- % Distance from 52-week Low

<center>NOTE: In the dataset the SPX is considered is treated as a stock with the symbol <b>^GSPC</b></center>
<br>

**Target**: *Outpreformed_Predicted_Next_Week* is the target column. This represents a shift of the Outperformed column back one date where the market is open (ie Monday $\rightarrow$ previous Friday). The outpreformed column on a given date represents the $\%\Delta$ between the Open price on the current day and the Close price 4 days later. 

The reason for the shift on the outperformed columns is because teh model most consider only data up to and including the previous Friday, and what is happening on the current monday is unknown. This way the result is shifted from the following Monday to the current Friday when the model is being tested. 

### Preprocessing
The first step in preprocessing was dropping all non-friday's from the dataset. This was done because we are making the prediction based on Friday data for the upcoming week.

The dataset was also particularly large leading to long training phases when using intensive models such as Calibrated CCV. In order to expedite run time and maintain accuracy of predictions the data was filtered by *Rolling 5-day Outperform*. This column represents the sum of the binary columns Daily Outperformed SP500, which checks if a stock beat out the SP500 on a given day. If stock had a Rolling 5-day Outperform<3 (ie outperformed the SP500 for $\frac35$ of the previous days) days it was dropped. This was orginally done after the model was run used and helped to improve predictive accuracy, but once this was discovered it was moved before learning. 

## Model Design
**To be added**