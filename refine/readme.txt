data.mat includes the CT images and CAMs in a format that could be easily processed by MATLAB.
chan_vese.m is the function deal with CT images.
main.m is the main program

Users can simply run main.m. The main program will take in the data, do the processing and show the result. The final result will be in a cell format. Each sub-cell in the cell is a patient and each matrix in a sub-cell is the segmentation result of one CT image for that patient.
