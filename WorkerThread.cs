using System.Linq; //for sum

namespace EmguStressTest
{
    class WorkerThread
    {
        private enum Operation
        {
            GAUSSIAN,
            PADDING,
            CONVERTTO,
            REGIONMASK,
            NOFELEMENTS
        }
        private static double[] CumulativeSum(double[] x)
        {
            var ret = new double[x.Length];
            double sum = 0;
            for (int i = 0; i < x.Length; ++i)
            {
                sum += x[i];
                ret[i] = sum;
            }
            return ret;
        }
        public WorkerThread(StressTest parent, string threadname, int seed)
        {
            //set things up for the operation selector
            if (true)
            {
                CumulativeProbabilityOfOperation = new double[(int)Operation.NOFELEMENTS];

                //this is the relative likelihoods of choosing those operations
                CumulativeProbabilityOfOperation[(int)Operation.GAUSSIAN] = 1;
                CumulativeProbabilityOfOperation[(int)Operation.PADDING] = 1;
                CumulativeProbabilityOfOperation[(int)Operation.CONVERTTO] = 1;
                CumulativeProbabilityOfOperation[(int)Operation.REGIONMASK] = 1;

                //accumulate
                CumulativeProbabilityOfOperation = CumulativeSum(CumulativeProbabilityOfOperation);
                //normalize
                double suminv = 1.0 / CumulativeProbabilityOfOperation.Last();
                for (int i = 0; i < CumulativeProbabilityOfOperation.Length; ++i)
                {
                    CumulativeProbabilityOfOperation[i] *= suminv;
                }
            }

            this.Parent = parent;
            this.Random = new System.Random(seed);
            this.Thread = new System.Threading.Thread(() => Entry(this));
            this.Thread.Name = threadname;
            this.Thread.Start();
        }
        // must be called from another thread
        internal void Abort()
        {
            this.Thread.Abort();
        }

        // must be called from another thread
        internal void Join()
        {
            this.Thread.Join();
        }

        private void Doit()
        {
            //we will repeatedly get an image, do an operation on it, then possibly
            //give it to someone else
            while (KeepRunning)
            {
                try
                {
                    //System.Console.WriteLine("I am a thread!");
                    Emgu.CV.UMat image1 = GetNextImage();

                    //perhaps randomly clone() it and/or set it to null or change it's type
                    Emgu.CV.UMat image2 = DoRandomHouseholding(image1);

                    //do some image operation
                    Emgu.CV.UMat image3 = OperateOnImage(image2);

                    //give it to someone else? or dispose it? or set it to null? who knows!
                    GetRidOfImage(image3);
                }
                catch (System.Threading.ThreadAbortException e)
                {
                    //System.Console.WriteLine($"oops, I'm aborted!");                    
                    this.StackTrace = e.StackTrace;
                    return;
                }
                catch (System.Exception e)
                {
                    //maybe something about mismatch in types or so, normal.
                }
                //signal that we are alive
                ++Counter;
            }
        }

        // randomly make up an image, or pick one from incoming
        private Emgu.CV.UMat GetNextImage()
        {

            double imagesource = Random.NextDouble();
            if (imagesource < ProbabilityOfMaking)
            {
                Emgu.CV.UMat image = MakeRandomImage();
                return image;
            }
            else
            {
                //take one from the input queue.
                return Parent.Pop();
            }
        }

        // randomly do something with the image
        private Emgu.CV.UMat DoRandomHouseholding(Emgu.CV.UMat image)
        {
            if (image == null)
            {
                return null;
            }
            double selector = Random.NextDouble();
            if (selector < 0.1)
            {
                //the input image will be garbage collected - for the finalizer to eventuall reap.
                return image.Clone();
            }
            if (selector < 0.2)
            {
                image.Dispose();
                return null;
            }
            return image;
        }

        // randomly operate on the given image
        private Emgu.CV.UMat OperateOnImage(Emgu.CV.UMat input)
        {
            if (input == null)
            {
                return null;
            }
            double selector = Random.NextDouble();

            //pick out the first element that is not less than selector
            int chosen = 0;
            for (int i = 0; i < CumulativeProbabilityOfOperation.Length; ++i)
            {
                if (CumulativeProbabilityOfOperation[i] <= selector)
                {
                    //still lower.
                }
                else
                {
                    //this is it!
                    chosen = i;
                    break;
                }
            }

            switch (chosen)
            {
                case (int)Operation.GAUSSIAN:
                    return Do_Gaussian(input);
                case (int)Operation.PADDING:
                    return Do_Padding(input);
                case (int)Operation.CONVERTTO:
                    return Do_ConvertTo(input);
                case (int)Operation.REGIONMASK:
                    return Do_RegionMask(input);
                default:
                    //weird. do nothing!
                    return input;
            }

            return input;
        }

        private void GetRidOfImage(Emgu.CV.UMat image)
        {
            if (image == null)
            {
                return;
            }
            double selector = Random.NextDouble();
            if (selector < 0.85)
            {
                //give it to the parent, for someone else to use
                Parent.Push(image);
                return;
            }

            //dispose it. but only maybe. this is to trigger the
            //finalizer.
            if (Random.NextDouble() < 0.8)
            {
                image.Dispose();
            }
        }

        private Emgu.CV.UMat Do_Gaussian(Emgu.CV.UMat input)
        {
            double sigmax = 1 + Random.NextDouble() * 2;
            double sigmay = 1 + Random.NextDouble() * 2;

            var ksize = System.Drawing.Size.Empty;
            Emgu.CV.UMat output;
            bool inplace = GetInPlace();
            if (inplace)
            {
                output = input;
            }
            else
            {
                output = new Emgu.CV.UMat();
            }
            Emgu.CV.CvInvoke.GaussianBlur(input, output, ksize, sigmax, sigmay);
            if(!inplace)
            {
                input.Dispose();
            }
            return output;
        }

        private Emgu.CV.UMat Do_Padding(Emgu.CV.UMat input)
        {
            double sigmax = 1 + Random.NextDouble() * 2;
            int top = Random.Next(0, 5);
            int bottom = Random.Next(0, 5);
            int left = Random.Next(0, 5);
            int right = Random.Next(0, 5);

            var output = new Emgu.CV.UMat();
            Emgu.CV.CvInvoke.CopyMakeBorder(input, output, top, bottom, left, right, Emgu.CV.CvEnum.BorderType.Reflect, new Emgu.CV.Structure.MCvScalar(Random.NextDouble()));
            input.Dispose();
            return output;
        }

        private Emgu.CV.UMat Do_ConvertTo(Emgu.CV.UMat input)
        {
            //since 0 and 1 are very common input to scaling, select them with much higher probability than other random values
            double scale = 1;
            double offset = 0;
            if (Random.Next(2) != 0)
            {
                scale = Random.NextDouble();
            }
            if (Random.Next(2) != 0)
            {
                offset = Random.NextDouble();
            }
            var output = new Emgu.CV.UMat();
            input.ConvertTo(output, MakeRandomFormat(), scale, offset);
            input.Dispose();
            return output;
        }

        private Emgu.CV.UMat Do_RegionMask(Emgu.CV.UMat input)
        {
            int x = Random.Next(0, input.Size.Width - 1);
            int y = Random.Next(0, input.Size.Height - 1);
            int width = Random.Next(1, input.Size.Width - x);
            int height = Random.Next(1, input.Size.Height - y);
            System.Drawing.Rectangle roi = new System.Drawing.Rectangle(x, y, width, height);
            if (true)
            {
                System.Diagnostics.Contracts.Contract.Assert(0 <= roi.Top && roi.Top <= input.Size.Height);
                System.Diagnostics.Contracts.Contract.Assert(0 <= roi.Bottom && roi.Bottom <= input.Size.Height);
                System.Diagnostics.Contracts.Contract.Assert(0 <= roi.Left && roi.Left <= input.Size.Width);
                System.Diagnostics.Contracts.Contract.Assert(0 <= roi.Right && roi.Right <= input.Size.Width);
            }

            Emgu.CV.UMat cvResult = new Emgu.CV.UMat(input.Size, input.Depth, input.NumberOfChannels);
            using (var inputInsideRoi = new Emgu.CV.UMat(input, roi))
            using (var resultInsideRoi = new Emgu.CV.UMat(cvResult, roi))
            {
                cvResult.SetTo(new Emgu.CV.Structure.MCvScalar(0));

                inputInsideRoi.CopyTo(resultInsideRoi);
              
            }
            input.Dispose();
            return cvResult;
        }

        private bool GetInPlace()
        {
            return Random.NextDouble() < ProbabilityOfInplace;
        }

        private static void Entry(WorkerThread c)
        {
            c.Doit();
        }

        private Emgu.CV.UMat MakeRandomImage()
        {
            int width = Random.Next(1, 2049);
            int height = Random.Next(1, 2049);
            var size = new System.Drawing.Size(width, height);
            var image = new Emgu.CV.UMat(size, MakeRandomFormat(), 1);
            image.SetTo(new Emgu.CV.Structure.MCvScalar(1));
            return image;
        }

        Emgu.CV.CvEnum.DepthType MakeRandomFormat()
        {
            int selector = Random.Next(1, 8);

            switch (selector)
            {

                case 1:
                    return Emgu.CV.CvEnum.DepthType.Cv8U;
                case 2:
                    return Emgu.CV.CvEnum.DepthType.Cv8S;
                case 3:
                    return Emgu.CV.CvEnum.DepthType.Cv16U;
                case 4:
                    return Emgu.CV.CvEnum.DepthType.Cv16S;
                case 5:
                    return Emgu.CV.CvEnum.DepthType.Cv32S;
                case 6:
                    return Emgu.CV.CvEnum.DepthType.Cv32F;
                case 7:
                    return Emgu.CV.CvEnum.DepthType.Cv64F;
                default:
                    throw new System.Exception("gah");
            }
        }
        //an atomic counter. one thread will read/write to it and the other read,
        //so incrementing it will work correctly.
        public System.Int32 Counter { get; private set; } = 0;
        public bool KeepRunning { get; set; } = true;
        public StressTest Parent { get; private set; }
        private System.Threading.Thread Thread { get; set; }
        private System.Random Random { get; set; }
        private double ProbabilityOfMaking { get; set; } = 0.2;
        private double ProbabilityOfInplace { get; set; } = 0.1;
        private double ProbabilityOfGpuOperation { get; set; } = 0.9;
        private double ProbabilityOfCpuOperation { get; set; } = 0.05;
        private double[] CumulativeProbabilityOfOperation { get; set; }
        public string StackTrace { get; private set; } = "";
    }
}
