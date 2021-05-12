using Encog.MathUtil.Randomize;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.ML.Train;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Training;
using Encog.Neural.Networks.Training.PSO;
using Encog.Util.Simple;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Encog_master
{
    class Program
    {
        //Input for the XOR function.
        public static double[][] XORInput = { new[] { 0.0, 0.0 }, new[] { 1.0, 0.0 }, new[] { 0.0, 1.0 }, new[] { 1.0, 1.0 } };

        //Ideal output for the XOR function.
        public static double[][] XORIdeal = { new[] { 0.0 }, new[] { 1.0 }, new[] { 1.0 }, new[] { 0.0 } };

        static void Main(string[] args)
        {
            //Create a basic training data set using the supplied data shown above
            IMLDataSet trainingSet = new BasicMLDataSet(XORInput, XORIdeal);

            //Create a simple feed forward network
            BasicNetwork network = EncogUtility.SimpleFeedForward(2, 2, 0, 1, false);

            //Create a scoring/fitness object
            ICalculateScore score = new TrainingSetScore(trainingSet);

            //Create a network weight initializer
            IRandomizer randomizer = new NguyenWidrowRandomizer();

            //Create the NN PSO trainer. This is our replacement function from backprop
            IMLTrain train = new NeuralPSO(network, randomizer, score, 20);

            //Train the application until it reaches an error rate of 0.01
            EncogUtility.TrainToError(train, 0.01);
            network = (BasicNetwork)train.Method;

            //Print out the results
            EncogUtility.Evaluate(network, trainingSet);

            Console.ReadKey();
        }
    }
}
