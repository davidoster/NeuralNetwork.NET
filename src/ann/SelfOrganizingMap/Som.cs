using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace ann.SelfOrganizingMap
{
    public class Som
    {

        int Height;
        int Width;
        int Radius;
        int Total;
        double LearningRate;
        SNode[] Nodes;
        int FvSize;
        int PvSize;

        public double Distance(SNode node1, SNode node2) => Math.Sqrt(Math.Pow((double)(node1.X - node2.X), 2) + Math.Pow((double)(node1.Y - node2.Y), 2));

        public Som(int height, int width, int fvSize, int pvSize)
        {
            Total = height * width;
            Height = height;
            Width = width;
            Radius = (height + width) / 2;
            LearningRate = 0.05;
            FvSize = fvSize;
            PvSize = pvSize;
            Nodes = new SNode[Total];

            for (int i = 0; i < Height; i++)
                for (int j = 0; j < Width; j++)
                    Nodes[i * Width + j] = new SNode(fvSize, pvSize, i, j);

        }

        public int BestMatch(double[] fvTarget)
        {
            var minimum = Math.Sqrt((double)(FvSize));
            var minimumIndex = 1;

            for (var i = 0; i < Total; i++)
            {
                var temp = fvDistance(Nodes[i].Fv, fvTarget);

                if (temp < minimum)
                {
                    minimum = temp;
                    minimumIndex = i;
                }
            }
            return minimumIndex;
        }


        public double fvDistance(double[] fv1, double[] fv2)
        {
            var temp = 0.0;

            for (var j = 0; j < FvSize; j++)
                temp = temp + Math.Pow(fv1[j] - fv2[j], 2);

            temp = Math.Sqrt(temp);
            return temp;
        }


        struct StackValue
        {
            public int k;
            public double[] fvTemp;
            public double[] pvTemp;
        }

        public void Train(int iterations, double[][] fvInputTrain, double[][] pvInputTrain)
        {
            if (fvInputTrain.Length != pvInputTrain.Length)
                throw new Exception("length of fvInputTrain should match pvInputTrain");


            var timeConstant = ((double)iterations) / Math.Log(((double)Radius));
            var radiusDecaying = 0.0;
            var lrd = 0.0; // learning rate decaying
            var influence = 0.0;

            var stack = new List<StackValue>();
            var length = fvInputTrain.Length;

            for (var i = 1; i < iterations + 1; i++)
            {
                radiusDecaying = ((double)Radius) * Math.Exp(-1.0 * i / timeConstant);
                lrd = LearningRate * Math.Exp(-1.0 * i / timeConstant);

                for (var j = 0; j < length; j++)
                {

                    var fvInput = fvInputTrain[j];
                    var pvInput = pvInputTrain[j];
                    var best = BestMatch(fvInput);

                    stack = new List<StackValue>();

                    for (var k = 0; k < Total; k++)
                    {

                        var dist = Distance(Nodes[best], Nodes[k]);

                        if (dist < radiusDecaying)
                        {
                            var fvTemp = new List<double>();
                            var pvTemp = new List<double>();

                            influence = Math.Exp((-1.0 * Math.Pow(dist, 2)) / (2 * radiusDecaying * ((double)i)));

                            // perform FV learning
                            for (var m = 0; m < FvSize; m++)
                            {
                                var adjustment = influence * lrd * (fvInput[m] - Nodes[k].Fv[m]);
                                fvTemp.Add(Nodes[k].Fv[m] + adjustment);
                            }

                            // perform PV learning
                            for (var m = 0; m < PvSize; m++)
                            {
                                var adjustment = influence * lrd * (pvInput[m] - Nodes[k].Pv[m]);
                                pvTemp.Add(Nodes[k].Pv[m] + adjustment);

                            }

                            stack.Add(new StackValue() { k = k, fvTemp = fvTemp.ToArray(), pvTemp = pvTemp.ToArray() });

                        }
                    }


                    // update nodes with new learned values
                    var stackLen = stack.Count();
                    for (var k = 0; k < stackLen; k++)
                    {
                        Nodes[stack[k].k].Fv = stack[k].fvTemp;
                        Nodes[stack[k].k].Pv = stack[k].pvTemp;
                    }


                }

            }
        }

        // Predict performs prediction for SOM.
        public double[] Predict(double[] fv) => Nodes[BestMatch(fv)].Pv;

        // PredictInt performs prediction for SOM and rounds resulting values to percentage.
        public int[] PredictInt(double[] fv)
        {
            var best = BestMatch(fv);
            var res = new List<int>();

            foreach (var item in Nodes[best].Pv)
                res.Add((int)(item * 100));

            return res.ToArray();
        }


    }
}
