import img2img

### ENTER INSPIRATION IMAGE NAME HERE ###
inspiration_image_location = ""  # only modify inside the quotes here

### ENTER PROMPT HERE
prompt = ""

### ENTER DESIRED LEVEL OF GENERATIVE ADDITION HERE 0 (none) to 1 (full)
desired_strength = 0.75


########################################################################
#################### DO NOT MODIFY BELOW HERE ##########################
########################################################################

img2img.generate(inspiration_img=inspiration_image_location,
                 prompt=prompt,
                 strength=desired_strength,
                 seed=i
                 )

