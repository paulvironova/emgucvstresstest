using System;

namespace EmguStressTest
{

    internal struct HangStatusSingle
    {
        //this pair represents the earliest known time when counter was that value.
        // in case the counter is reported to be the same value again, the timestamp
        // will remain at the last value
        private int lastcounter { get; set; }

        public TimeSpan When { get; private set; }

        public void refresh(int value, System.Diagnostics.Stopwatch watch)
        {
            if(value==lastcounter)
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
            string hangstring="";
            for(int i=0; i<Nthreads_;++i)
            {
                TimeSpan dt = now - status_[i].When;
                hangstring += System.FormattableString.Invariant($"{i}:{dt.TotalSeconds:F1}  ");                
            }
            System.Console.WriteLine(hangstring);
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

        // thread safe wrt. Pop
        internal void Push(Emgu.CV.UMat image)
        {
            if (image != null)
            {
                if (ImageQueue.Count < 1000)
                {
                    ImageQueue.Enqueue(image);
                } else
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
            if(ImageQueue.TryDequeue(out ret))
            {
                return ret;
            }
            return null;
        }

        private void Run()
        {
            var threads= new System.Collections.Generic.List<WorkerThread>();
            for(int i=0; i<Nthreads_; ++i)
            {
                var wt = new WorkerThread(this,$"WT {i}",1234+i);
                threads.Add(wt);
            }
            var hangstatus = new HangStatusGroup(Nthreads_);
            while (true)
            {
                //wait a while before asking the threads how it goes.                
                System.Threading.Thread.Sleep(1000);
                //ask the threads
                for (int i = 0; i < Nthreads_; ++i)
                {
                    hangstatus.updateThread(i, threads[i].Counter);
                }
                hangstatus.showStatus();
            }
            //make sure *all* the threads make progress, otherwise they are in a deadlock.
            //this works like a watchdog.
            var lastvalues=new int[Nthreads_];
            var currentvalues = new int[Nthreads_];

            var thread_is_stuck = new bool[Nthreads_];
            bool anyone_is_stuck = false;
            while (!anyone_is_stuck)
            {
                for (int i = 0; i < 10; ++i)
                {
                    System.Threading.Thread.Sleep(WatchDogTimeoutMs/10);
                    System.Console.WriteLine($"ImageQueue.Count={ImageQueue.Count}");
                }
                for (int i = 0; i < Nthreads_; ++i)
                {
                    currentvalues[i] = threads[i].Counter;
                    if (lastvalues[i] == currentvalues[i])
                    {
                        //this thread is stuck.
                        System.Console.WriteLine($"Thread {i} does not make progress (counter={currentvalues[i]}).");
                        //throw new System.Exception($"Watchdog - thread {i} has not make progress for at least {WatchDogTimeoutMs} ms.");
                        //anyone_is_stuck = true;
                        thread_is_stuck[i] = true;
                    }
                   
                    lastvalues[i] = currentvalues[i];
                }
                if (!anyone_is_stuck)
                {
                    System.Console.WriteLine($"All threads are making progress. First thread has made {currentvalues[0]} iterations.");
                }
            }

            for (int i = 0; i < Nthreads_; ++i)
            {
                if(!thread_is_stuck[i])
                {
                    threads[i].KeepRunning = false;
                }
            }
            while (true)
            {
                System.Threading.Thread.Sleep(1000);
                System.Console.WriteLine($"Watchdog - all threads except the stuck one should now exit, suggesting to break in the debugger");
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
