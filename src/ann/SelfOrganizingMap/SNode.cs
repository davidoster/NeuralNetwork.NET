using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ann.SelfOrganizingMap
{
    public class SNode
    {

        public SNode(int fvSize, int pvSize, int y, int x)
        {
            X = x;
            Y = y;
            Fv = new double[fvSize];
            Pv = new double[pvSize];

            var rand = new Random();

            for (int i = 0; i < fvSize; i++)
                Fv[i] = rand.NextDouble();

            for (int i = 0; i < pvSize; i++)
                Pv[i] = rand.NextDouble();

        }
        public double[] Fv { get; set; }
        public double[] Pv { get; set; }
        public int X { get; set; }
        public int Y { get; set; }


        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append("Node FV [");
            sb.Append(string.Join(",", Fv));
            sb.Append("] PV [");
            sb.Append(string.Join(",", Pv));

            return sb.ToString();
        }

    }



}
