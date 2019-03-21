using System;

namespace EmguStressTest
{

    internal struct HangStatusSingle
    {
        //this pair represents the earliest known time when counter was that value.
        // in case the counter is reported to be the same value again, the timestamp
        // will remain at the last value
        public int lastcounter { get; private set; }

        public TimeSpan When { get; private set; }

        public void refresh(int value, System.Diagnostics.Stopwatch watch)
        {
            if (value == lastcounter)
            {
                //no update.
                return;
            }
            //counter has moved. update the timestamp.
            lastcounter = value;
            When = watch.Elapsed;
        }
    }
    internal class HangStatusGroup
    {
        public HangStatusGroup(int nthreads)
        {
            Nthreads_ = nthreads;
            status_ = new HangStatusSingle[Nthreads_];
            clock_ = new System.Diagnostics.Stopwatch();
            clock_.Start();
        }
        public void updateThread(int index, int value)
        {
            status_[index].refresh(value, clock_);
        }
        public void showStatus()
        {
            var now = clock_.Elapsed;
            string hangstring = "";
            long total = 0;
            for (int i = 0; i < Nthreads_; ++i)
            {
                TimeSpan dt = now - status_[i].When;
                hangstring += System.FormattableString.Invariant($"{dt.TotalSeconds,5:F1}  ");
                total += status_[i].lastcounter;
            }
            System.Console.WriteLine($"{total}" + hangstring);
        }
        public TimeSpan AgeOfUpdate(int i)
        {
            var now = clock_.Elapsed;
            TimeSpan dt = now - status_[i].When;
            return dt;
        }
        public TimeSpan OldestUpdate()
        {
            var ret = TimeSpan.MinValue;
            var now = clock_.Elapsed;
            string hangstring = "";
            long total = 0;
            for (int i = 0; i < Nthreads_; ++i)
            {
                TimeSpan dt = now - status_[i].When;
                if (dt > ret)
                {
                    ret = dt;
                }
            }
            return ret;
        }

        private int Nthreads_ = -1;
        private HangStatusSingle[] status_;
        private System.Diagnostics.Stopwatch clock_;
    }

    public class StressTest
    {

        public int Nthreads_ { get; private set; } = System.Environment.ProcessorCount;

        public int WatchDogTimeoutMs { get; private set; } = 15000;

        private System.Collections.Concurrent.ConcurrentQueue<Emgu.CV.UMat> ImageQueue { get; set; } = new System.Collections.Concurrent.ConcurrentQueue<Emgu.CV.UMat>();

        public TimeSpan HangtimeLimit { get; private set; } = TimeSpan.FromSeconds(200);

        // thread safe wrt. Pop
        internal void Push(Emgu.CV.UMat image)
        {
            if (image != null)
            {
                if (ImageQueue.Count < Nthreads_ * 2)
                {
                    ImageQueue.Enqueue(image);
                }
                else
                {
                    //we are reaching into memory limits, so dispose it to prevent
                    //exhausting the ram
                    image.Dispose();
                }
            }
        }
        // thread safe wrt. Push
        internal Emgu.CV.UMat Pop()
        {
            Emgu.CV.UMat ret = null;
            if (ImageQueue.TryDequeue(out ret))
            {
                return ret;
            }
            return null;
        }

        private void Run()
        {
            var threads = new System.Collections.Generic.List<WorkerThread>();
            for (int i = 0; i < Nthreads_; ++i)
            {
                var wt = new WorkerThread(this, $"WT {i}", 1234 + i);
                threads.Add(wt);
            }
            var hangstatus = new HangStatusGroup(Nthreads_);

            bool keeprunning = true;
            while (keeprunning)
            {
                //wait a while before asking the threads how it goes.                
                System.Threading.Thread.Sleep(1000);
                //ask the threads
                for (int i = 0; i < Nthreads_; ++i)
                {
                    hangstatus.updateThread(i, threads[i].Counter);
                }
                hangstatus.showStatus();

                for (int i = 0; i < Nthreads_; ++i)
                {
                    TimeSpan age = hangstatus.AgeOfUpdate(i);
                    if (age > HangtimeLimit)
                    {
                        System.Console.WriteLine($"Aborting thread {i} because it appears stuck.");
                        threads[i].Abort();
                        keeprunning = false;
                    }
                }
            }

            //stop all threads normally
            System.Console.WriteLine($"stopping all threads gently...");
            for (int i = 0; i < Nthreads_; ++i)
            {
                threads[i].KeepRunning = false;
            }

            //join them
            for (int i = 0; i < Nthreads_; ++i)
            {
                threads[i].Join();
                System.Console.WriteLine($"Thread {i} has stacktrace: {threads[i].StackTrace}");
            }
        }


        static void Main(string[] args)
        {
            //this does not work
            //System.Environment.SetEnvironmentVariable("OPENCV_OPENCL_DEVICE", ":GPU:0");
            System.Console.WriteLine($"todiloo! Emgu GPU is {Emgu.CV.Ocl.Device.Default.Name}");

            //setup a program object
            var st = new StressTest();

            st.Run();
        }
    }
}
