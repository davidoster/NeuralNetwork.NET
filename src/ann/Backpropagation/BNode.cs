using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace ann.Backpropagation
{
    public class BNode
    {

        public BNode(int wCount)
        {
            Weights = new double[wCount];
        }

        public double Threshold { get; set; }

        public double[] Weights { get; set; }


        public double ActivationValue { get; set; }

        public double Error { get; set; }
    }



}
