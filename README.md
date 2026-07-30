Files in project:

UNet.py - script with UNet class used by both the old pipeline and the new pipeline

old_pipeline.py - Performs data inpainting based on either rf or IQ data. Works with different central frequencies. Able add new columns to the image

old_load_data.py - loading data and preprocessing for the old pipeline

new_pipeline.py - Performs data inpainting based on IQ data, works with different beam angles

preprocess.py - preprocesses the data for the new pipeline

data_loader.py - designed to not overflow the memory while using large datasets with new pipeline

INR.py - script with SIREN INR implementation

INR_program.py - performs data inpainting using SIREN

PINN_program.py - performs data inpainting using SIREN supplemented by loss coming form the wave equation

plots.py - handles plots for all the other files
