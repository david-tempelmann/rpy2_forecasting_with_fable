# Databricks notebook source
# MAGIC %md ## A forecasting example in R
# MAGIC We simply use the example included as part of the [fable](https://fable.tidyverts.org/#example) package.
# MAGIC This example demonstrates how to use fable to forecast the turnover of deparment stores Australia using
# MAGIC ETS, Arima and seasonal naive models.
# MAGIC 
# MAGIC Make sure to install the required R packages, e.g. as cluster libraries.

# COMMAND ----------

# MAGIC %r
# MAGIC library(fpp3)
# MAGIC aus_retail %>%
# MAGIC   filter(
# MAGIC     State %in% c("New South Wales", "Victoria"),
# MAGIC     Industry == "Department stores"
# MAGIC   ) %>% 
# MAGIC   model(
# MAGIC     ets = ETS(box_cox(Turnover, 0.3)),
# MAGIC     arima = ARIMA(log(Turnover)),
# MAGIC     snaive = SNAIVE(Turnover)
# MAGIC   ) %>%
# MAGIC   forecast(h = "2 years") %>% 
# MAGIC   autoplot(filter(aus_retail, year(Month) > 2010), level = 80)

# COMMAND ----------

# MAGIC %r
# MAGIC aus_retail

# COMMAND ----------

# MAGIC %md
# MAGIC ## Not let's run the same example from a python environment
# MAGIC We are now going to use rpy2 to run the exact same example shown above from within Python.
# MAGIC Be careful with notebook-scoped libraries as these will be installed in virtual environments local to the notebooks. However, the path to the local R environment is not known to the Python environment from where we run R. 
# MAGIC Hence, to work with Rpy2 the required R-libraries need
# MAGIC to be installed from within Python or as cluster-scoped libraries.

# COMMAND ----------

# DBTITLE 1,Install Rpy2
# MAGIC %pip install rpy2

# COMMAND ----------

# DBTITLE 1,Python Imports
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.lib.dplyr import DataFrame
from rpy2.robjects import rl

# COMMAND ----------

# DBTITLE 1,Import R packages into Python Objects
base = importr("base")
tsibble = importr("tsibble")
fabletools = importr("fabletools")
fpp3 = importr("fpp3")
tsibbledata = importr("tsibbledata")
distributional = importr("distributional")

# COMMAND ----------

# DBTITLE 1,Convert Australian retail data from R to Python for further manipulation in Python
# Rpy2's conversion to pandas works for R data frames, however, the data is an R tsibble
# Hence, we convert from R's tsibble format to R data frame
aus_retail_r_df = base.as_data_frame(ro.r("tsibbledata::aus_retail"))

# R has a date type while pandas only has datetimes so we need
# to convert to datetime to make sure conversion to pandas works
aus_retail_r_df = DataFrame(aus_retail_r_df).mutate(Month=rl("lubridate::as_datetime(Month)"))

# Now convert to pandas using Rpy2's localconverter
with localconverter(ro.default_converter + pandas2ri.converter):
  aus_retail_pdf = ro.conversion.rpy2py(aus_retail_r_df)

# COMMAND ----------

# DBTITLE 1,Manipulate the data in Python - Filter States and Industries
aus_retail_pdf = aus_retail_pdf[
  aus_retail_pdf.State.isin(["New South Wales", "Victoria"]) & 
  aus_retail_pdf.Industry.isin(["Department stores"])]
aus_retail_pdf

# COMMAND ----------

# DBTITLE 1,Convert manipulated data back to an R tsibble
with localconverter(ro.default_converter + pandas2ri.converter):
  rdf = ro.conversion.py2rpy(aus_retail_pdf)

# we need to convert the datetime back to tsibble's yearmonth format
# we also define the time index and (primary) keys for the time series data
rts = tsibble.as_tsibble(
  DataFrame(rdf).mutate(Month=rl("tsibble::yearmonth(Month)")),
  index="Month",
  key=ro.StrVector(['State', 'Industry']),
)

# COMMAND ----------

# DBTITLE 1,Print R representation
print(rts)

# COMMAND ----------

# DBTITLE 1,Train 3 R forecasting models
# the function rl creates unevaluated R language objects which are
# in this case consumed by the model function
r_models = fabletools.model(
  rts, 
  rl("ets=ETS(box_cox(Turnover, 0.3))"), 
  rl("arima = ARIMA(log(Turnover))"), 
  rl("snaive = SNAIVE(Turnover)")
)

# COMMAND ----------

# DBTITLE 1,Forecast 2 years and extract mean as well as prediction intervals
r_forecast = fabletools.forecast(r_models, h="2 years")
# get 95% prediction interval
fcst_dist = distributional.hilo(r_forecast, level=95.0)
fcst_dist = fabletools.unpack_hilo(fcst_dist, "95%")
# select relevant columns
fcst_r_df = base.as_data_frame(DataFrame(fcst_dist).select("State", "Industry", ".model", ".mean", "95%_lower", "95%_upper"))
# convert from tsibbles yearmonth to datetime which can be converted to pandas
r_forecast_date = DataFrame(fcst_r_df).mutate(Month=rl("lubridate::as_datetime(Month)"))

# convert to pandas
with localconverter(ro.default_converter + pandas2ri.converter):
  forecast_pdf = ro.conversion.rpy2py(r_forecast_date)

# COMMAND ----------

forecast_pdf

# COMMAND ----------


