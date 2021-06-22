## Sox
### sox(Sound eXchange)
SoX is a cross-platform command line audio utility tool that works on Linux, Windows and MacOS. It is very helpful in the following areas while dealing with audio and music files.  
* Audio File Converter
* Editing audio files  
* Changing audio attributesAdding audio effects
* Plus lot of advanced sound manipulation features  
In general, audio data is described by following four  characteristics:  
**Rate** – sample rate,  **Data size** – The precision the data is stored in.  For example, 8/16 bits
**Data encoding, Channels** – How many channels are contained in the audio data.  For example, Stereo 2 channels  
**1.Combine Multiple Audio Files to Single File**  
adds first_part.wav and second_part.wav leaving the result in whole_part.wav  
```
sox -m first_part.wav second_part.wav whole_part.wav
```
**2.Extract Part of the Audio File**  
```
#extract first 10 seconds from input.wav and stored it in output.wav
sox input.wav output.wav trim 0 10
```