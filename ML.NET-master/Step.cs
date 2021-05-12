using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using Microsoft.ML;
using Microsoft.ML.AutoML;

namespace ML.NET_master
{
    class Step
    {
        public void run()
        {
            // Use the AutoMated ML API
            MLContext mLContext = new MLContext();
            IDataView trainDataView = mLContext.Data.LoadFromTextFile<SentimentIssue>("my-data-file.csv", hasHeader: true);

            //Create experiment settings for the determined ML task type:
            //Binary Classification
            //var experimentSettings = new BinaryExperimentSettings();

            //Multiclass Classification
            //var experimentSettings2 = new MulticlassExperimentSettings();

            //Regression
            //var experimentSettings3 = new RegressionExperimentSettings();

            //Recommendation
            //var experimentSetting4 = new RecommendationExperimentSettings();

            //Create experiment settings
            var cts = new CancellationTokenSource();
            var experimentSettings = new RegressionExperimentSettings();
            experimentSettings.MaxExperimentTimeInSeconds = 3600;
            experimentSettings.CancellationToken = cts.Token;

            //Create an experiment
            RegressionExperiment experiment = mLContext.Auto().CreateRegressionExperiment(experimentSettings);

            //Run the experiment
            IDataView trainingDataView = null;
            string LabelColumnName = string.Empty;
            ExperimentResult<Microsoft.ML.Data.RegressionMetrics> experimentResult = experiment.Execute(trainingDataView, LabelColumnName);

            //Training modes
            experiment.Execute(trainDataView);

            IDataView validationDataView = null;
            experiment.Execute(trainDataView, validationDataView);

            //Explore model metrics
            Microsoft.ML.Data.RegressionMetrics metrics = experimentResult.BestRun.ValidationMetrics;
            Console.WriteLine($"R-Squared: {metrics.RSquared:0.##}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError:0.##}");
        }
    }
    internal class SentimentIssue
    {
    }
}
