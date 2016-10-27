using ann.Backpropagation;
using ann.SelfOrganizingMap;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using static System.Console;

namespace ann.TestApp
{
    public class Program
    {

        public static double[] ParseIntToBinaryInput(int inputSize, int value) => Convert.ToString(value, 2).PadLeft(inputSize, '0').ToCharArray().Select(e => double.Parse(e.ToString())).ToArray();



        public static void Main(string[] args)
        {
            SOMPattern();
            BackpropPrimes();


        }


        private static void SOMPattern()
        {
            WriteLine("\n-- SOM init");

            var som = new Som(12, 12, 10, 3);

            // training patterns
            var data = new double[][] {
                new double[] {0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0},
                new double[] {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9},
                new double[] {0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1},
            };

            var result = new double[][] {
                new double[] {1.0, 0.0, 0.0},
                new double[] {0.0, 1.0, 0.0},
                new double[] {0.0, 0.0, 1.0},
            };

            var test = new double[][] {
                new double[]{0.9, 0.8, 0.3, 0.4, 0.4, 0.5, 0.4, 0.3, 0.2, 0.4} ,
                new double[]{0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8} ,
                new double[]{0.1, 0.2, 0.3, 0.4, 0.6, 0.6, 0.4, 0.3, 0.2, 0.1},
            };

            WriteLine("Training...");
            som.Train(5000, data, result);

            // test
            var expected = som.PredictInt(test[0]);
            if (expected[0] < 85)
                throw new Exception($"expected [0] to be above 85% got {expected[0]}");

            expected = som.PredictInt(test[1]);
            if (expected[1] < 85)
                throw new Exception($"expected [1] to be above 85% got {expected[1]}");

            expected = som.PredictInt(test[2]);
            if (expected[2] < 85)
                throw new Exception($"expected [2] to be above 85% got {expected[2]}");
            
            // simple parser array to string
            Func<double[], string> p1 = (e => string.Join(",", e));
            Func<int[], string> p2 = (e => string.Join(",", e));

            // show results
            WriteLine($"{p1( data[0] )}, {p2(som.PredictInt(data[0]) )}");
            WriteLine($"{p1( data[1] )}, {p2(som.PredictInt(data[1]) )}");
            WriteLine($"{p1( data[2] )}, {p2(som.PredictInt(data[2]) )}");

            WriteLine("Running predictions...");
            WriteLine($"{p1( test[0] )}, {p2( som.PredictInt(test[0]) )}");
            WriteLine($"{p1( test[1] )}, {p2( som.PredictInt(test[1]) )}");
            WriteLine($"{p1( test[2] )}, {p2( som.PredictInt(test[2]) )}");
        }

        private static void BackpropPrimes()
        {

            int[] primes = new int[] {  2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                                    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
                                    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
                                    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
                                    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
                                    233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
                                    283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
                                    353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
                                    419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
                                    467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
                                    547, 557, 563, 569, 571, 577, 587, 593, 599, 601,
                                    607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
                                    661, 673, 677, 683, 691, 701, 709, 719, 727, 733,
                                    739, 743, 751, 757, 761, 769, 773, 787, 797, 809,
                                    811, 821, 823, 827, 829, 839, 853, 857, 859, 863,
                                    877, 881, 883, 887, 907, 911, 919, 929, 937, 941,
                                    947, 953, 967, 971, 977, 983, 991, 997
                                };


            WriteLine("\n-- Backprop init");
            // helper map for checking if it is a prime or not
            var checkPrimes = new Dictionary<int, bool>();
            foreach (var n in primes)
                checkPrimes.Add(n, true);


            // size of our network input is 2 power of 10 = 1024 and is enough for primes up to 1000
            const int inputSize = 10;

            // training data is a slice with 1000 prime numbers converted into 1 and 0
            // and outputs are 1 if input is prime number
            var tr = new List<TrainingData>();

            for (var i = 0; i < 1000; i++)
            {
                var data = new TrainingData()
                {
                    Input = new double[inputSize],
                    Output = new double[1]
                };

                var binary = ParseIntToBinaryInput(inputSize, i);

                data.Input = binary;
                data.Output[0] = checkPrimes.ContainsKey(i) && checkPrimes[i] ? 1 : 0;

                tr.Add(data);

            }

            // input layer has 10 neurons, hidden has 19 and output has 1
            var nn = new BackpropNet(10, 19, 1);
            nn.Train(5000, tr.ToArray());

            // test if network has been trained	and can recognize prime numbers
            //fmt.Println("Running predictions")
            var errorCount = 0;
            var errors = new Dictionary<int,bool>();


            for (var i = 0; i < 1000; i++)
            {

                var input = ParseIntToBinaryInput(inputSize, i);
                var res = nn.PredictInt(input);

                if ((checkPrimes.ContainsKey(i) && checkPrimes[i] && res[0] == 0))
                {
                    errors.Add(i, false);
                    errorCount++;
                }
                else if ( (!checkPrimes.ContainsKey(i) || !checkPrimes[i]) && res[0] == 1)
                {
                    errors.Add(i, true);
                    errorCount++;
                }

            }

            if (errorCount > 40)
                throw new Exception($"total errors should be around 14 to 28, got errors {errorCount}");

            WriteLine($"Total errors {errorCount}:\n");
            foreach (var item in errors)
                WriteLine($"{item.Key}={item.Value}");

            Read();
        }



    }





}
