# Background
I am archiving comic books from physical to digital because I do not have the room to store all of my physical comics. I have a digital archive of comics I keep in cbz format. I want to scan the remainder of my comics into cbz format. The workflow to do this has been daunting and I am looking for ways to speed it up. It has been daunting because there are several time-consuming elements.

1. Scanning the physical comic to my computer
2. Detecting the bage bounds for the scanned page
3. Rotating the images when needed
4. De-skewing the images so they are square and consistent
5. Aligning all of the images and ensuring they are the same size
6. Applying a black background behind the images to correct for any images that are not perfectly algined with the others
7. Gwtting all of the comic images named and numbered in order
8. Filling in the comic XML data
9. Creating the cbz file

There is also an added step of converting the images so they are optimized (max quality with low file size, lossless) but I forget when I have done that in the past - I think I used imagemagick or something like it.

# What I want to do
I want to see if I can create a Python application that will do as much of this as possible for me. In the past I have used ScanTailorAdvanced (https://github.com/4lex4/scantailor-advanced) to do this, but it's a lenghty and complex proces. Perhaps it's codebase will be a good starting point.

I want the workflow to be like this.

1. I scan all of the pages into a directory per episode of the comic. The images will be 300 dpi JPG files with a bounding box larger than the scanner. 
2. Python script then performs all processing of the images (rotate, deskew, align, convert, etc.)
3. Python script then puts all image into a folder, creates the XML data, and creates CBZ file
4. Python sctipt then performs a QC check, looking for bad pages, missing pages, etc.

# Background materials
There is a folder titled `good-examples` with two CBZ files (and the decompressed contents) that are examples I created that serve as good examples of outputs.

There is a folder titled `raw-scans` that has a raw scan I just did of a new comic.

# Outputs
One or more Python scripts that perform the above. 
