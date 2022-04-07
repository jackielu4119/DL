### Hi there 👋
=======================================================================

Deduction Learning code for blood glucose prediction from PPG signal

                           Version 0.1
--------------------------------------------------------------------

This code consists N parts:

1. ppg_analy.py:
   The PPG signal preprocessing and analysis code.

2. XXXXXX 

Currently we have ppg_analy.py code ready for posting in github, and
the other parts are still under checking and preparation. They will
be posted as soon as everything is ready, before August, 2022.

Usage:

- ppg_analy.py:

  To use this code, the system should install python (>= 3.5), python
  modules numpy, scipy, and gnuplot.

  This code reads the index file "sub_info*.txt", from which find the
  PPG signal data files, and proceed the signal analysis. The PPG signals
  consist Left hand, Right hand, each measured with IR and RD. The code
  parameters are configurable through the config file "ppg_input.txt".
  The code outputs the following data files:

  - *_signal.txt:
    The denoised AC and DC singals of the raw PPG signal.

  - *_minmax.txt:
    The indices of pulse separation of the PPG AC signal.

  - *_peak.txt, *_peaklist.txt, *_dA1res.txt:
    The morphological features of pulse waveform extracted from the PPG
    AC signal.

  - *_sigIRFT.txt, *_sigRDFT.txt:
    The Fourier transformed data of the PPG AC signal.

  - *_step.txt:
    The summarized list of extracted morphological features.

  To run this code, please put "ppg_analy.py" and "ppg_input.txt" in the
  same working directory, and run the command:

        python3 ./ppg_analy.py <full path of sub_info file>



Authors:
   Wei-Ru Lu(a), Wen-Tse Yang(a,b), Justin Chiu(a), Tung-Han Hsieh(a),
   Fu-Liang Yang(a)

   (a) Research Center for Applied Sciences, Academia Sinica.
   (b) Department of Biomechatronics Engineering, National Taiwan University.

=======================================================================