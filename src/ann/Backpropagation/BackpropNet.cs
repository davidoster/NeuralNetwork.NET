using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace ann.Backpropagation
{
    public class BackpropNet
    {
        BNode[] Input;
        BNode[] Hidden;
        BNode[] Output;
        double lhRate; // learning rate of the hidden layer
        double loRate; // learning rate of the output layer
        double[] netInput;
        double[] desiredOut;

        private double Sigmoid(double x) => 1 / (1 + Math.Exp(-x)); // sigmoid helper function


        public BackpropNet(int inCount, int hideCount, int outCount)
        {
            lhRate = 0.15;
            loRate = 0.2;
            Input = new BNode[inCount];
            Hidden = new BNode[hideCount];
            Output = new BNode[outCount];

            var rand = new Random();

            for (var i = 0; i < inCount; i++)
            {
                Input[i] = new BNode(hideCount);

                for (var j = 0; j < hideCount; j++)
                {
                    Input[i].Weights[j] = rand.NextDouble() - 0.49999;

                }
            }

            for (var i = 0; i < hideCount; i++)
            {
                Hidden[i] = new BNode(outCount);

                for (var j = 0; j < outCount; j++)
                    Hidden[i].Weights[j] = rand.NextDouble();
            }

            for (var i = 0; i < outCount; i++)
            {
                Output[i] = new BNode(0);
            }

            for (var i = 0; i < Hidden.Length; i++)
            {
                Hidden[i].Threshold = rand.NextDouble();
            }


            for (var i = 0; i < Output.Length; i++)
            {
                Output[i].Threshold = rand.NextDouble();
            }


        }

        public void Train(int iterations, TrainingData[] data)
        {
            var inputLen = Input.Length;
            var outputLen = Output.Length;

            for (var i = 0; i < iterations; i++)
            {
                foreach (var tr in data)
                {
                    if (inputLen != tr.Input.Length)
                        throw new Exception($"expected training data input length {inputLen} got {tr.Input.Length}");

                    if (outputLen != tr.Output.Length)
                        throw new Exception($"expected traing data output length  {outputLen} got {tr.Output.Length}");

                    netInput = tr.Input;
                    desiredOut = tr.Output;
                    TrainOnePattern();
                }
            }

        }

        public void TrainOnePattern()
        {
            calcActivation();
            calcErrorOutput();
            calcErrorHidden();
            calcNewThresholds();
            calcNewWeightsHidden();
            calcNewWeightsInput();
        }


        public void SetLearningRates(double lhRate, double loRate)
        {
            this.lhRate = lhRate;
            this.loRate = loRate;
        }

        public void calcActivation()
        {
            // a loop to set the activations of the hidden layer
            for (var h = 0; h < Hidden.Length; h++)
                for (var i = 0; i < Input.Length; i++)
                    Hidden[h].ActivationValue += netInput[i] * Input[i].Weights[h];

            // calculate the output of the hidden
            for (var h = 0; h < Hidden.Length; h++)
            {
                Hidden[h].ActivationValue += Hidden[h].Threshold;
                Hidden[h].ActivationValue = Sigmoid(Hidden[h].ActivationValue);
            }

            // a loop to set the activations of the output layer
            for (var o = 0; o < Output.Length; o++)
                for (var h = 0; h < Hidden.Length; h++)
                    Output[o].ActivationValue += Hidden[h].ActivationValue * Hidden[h].Weights[o];


            // calculate the output of the output layer
            for (var o = 0; o < Output.Length; o++)
            {
                Output[o].ActivationValue += Output[o].Threshold;
                Output[o].ActivationValue = Sigmoid(Output[o].ActivationValue);
            }

        }


        private void calcNewWeightsInput()
        {
            for (var i = 0; i < netInput.Length; i++)
            {
                var temp = netInput[i] * lhRate;

                for (var h = 0; h < Hidden.Length; h++)
                {
                    Input[i].Weights[h] += temp * Hidden[h].Error;

                }
            }
        }

        private void calcNewWeightsHidden()
        {
            for (var h = 0; h < Hidden.Length; h++)
            {
                var temp = Hidden[h].ActivationValue * loRate;

                for (var o = 0; o < Output.Length; o++)
                {
                    Hidden[h].Weights[o] += temp * Output[o].Error;

                }
            }
        }

        private void calcNewThresholds()
        {
            // computing the thresholds for next iteration for hidden layer
            for (var h = 0; h < Hidden.Length; h++)
            {
                Hidden[h].Threshold += Hidden[h].Error * lhRate;
            }

            // computing the thresholds for next iteration for output layer
            for (var o = 0; o < Output.Length; o++)
            {
                Output[o].Threshold += Output[o].Error * loRate;
            }


        }

        private void calcErrorHidden()
        {
            for (var h = 0; h < Hidden.Length; h++)
            {
                for (var o = 0; o < Output.Length; o++)
                {
                    Hidden[h].Error += Hidden[h].Weights[o] * Output[o].Error;
                }
                Hidden[h].Error *= Hidden[h].ActivationValue * (1 - Hidden[h].ActivationValue);
            }
        }

        private void calcErrorOutput()
        {
            for (var o = 0; o < Output.Length; o++)
            {
                Output[o].Error = Output[o].ActivationValue * (1 - Output[o].ActivationValue) *
                    (desiredOut[o] - Output[o].ActivationValue);

            }
        }

        public double calcTotalError()
        {
            double temp = 0.0;
            for (var o = 0; o < Output.Length; o++)
            {
                temp += Output[o].Error;
            }
            return temp;
        }

        // Predict calculates network output based on provided input, returns raw float64 activation value.
        public double[] Predict(double[] input)
        {
            netInput = input;
            calcActivation();
            var outp = new double[Output.Length];

            for (int i = 0; i < Output.Length; i++)
                outp[i] = Output[i].ActivationValue;

            return outp;

        }


        // PredictInt calculates network output based on provided input, this is main method to call after Train.
        public int[] PredictInt(double[] input)
        {
            netInput = input;
            calcActivation();
            var outp = new int[Output.Length];


            for (int i = 0; i < Output.Length; i++)
            {
                var node = Output[i];
                if (node.ActivationValue > 0.5)
                    outp[i] = 1;
            }

            return outp;
        }


    }
}
