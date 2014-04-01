#!/usr/bin/python -tt
import numpy as np
import matplotlib.pyplot as plt
import png
import os
from PIL import Image

from py_utils.results.metric import Metric
from py_utils.results.defaults import DEFAULT_SLICE,DEFAULT_IMAGE_EXT

class OutputMontage(Metric):
    """
    Class for outputting and image or volume, and allowing a redraw/update.
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for OutputMontage. See :func:`py_utils.results.metric.Metric.__init__`.

        :param ps_parameters: :class:`py_utils.parameter_struct.ParameterStruct` object which should have 'slice', 'outputformat', and 'outputdirectory' in the :param:`str_section` heading.
        
        """
        super(OutputMontage,self).__init__(ps_parameters,str_section)
        self.output_directory = self.get_val('outputdirectory',False)
        self.montage_im_file = self.get_val('montageimagefile',False)
        self.montage_html_file = self.get_val('montagehtmlfile',False)
        self.thumbnail_width = self.get_val('thumbnailwidth',True)
        self.thumbnail_height = self.get_val('thumbnailheight',True)
        self.thumbnail_columns = self.get_val('thumbnailcolumns',True)
        self.thumbnail_rows = self.get_val('thumbnailrows',True)

        self.print_values = 0 #we never want to print array data...
        self.has_csv = False #we can't save these to csv format like other metrics
        
    def update(self,dict_in):
        """Takes a 2D or 3D image or volume . If a volume, display the :attr:`self.slice` if specified, otherwise display volume using Myavi. Aggregate images/slices into a volume to view the reconstruction later, or save volumes
        :param dict_in: Input dictionary which contains the referenece to the image/volume data to display and record. 
        """
        super(OutputMontage,self).update()

    def plot(self): pass

    def save(self,strPath='/home/outputimage'):
        """Take the parameters in dict_in to write the montage image and all of the submimages
        'output_directory': the output directory 
        'montage_jpg': the output jpg filename
        'montage_html': the output html filename
        'thumbnail_columns': the number of columns in the montage    
        'thumbnail_rows': the number of rows in the montage    
        'ls_images': list of np arrays (of images)
        'ls_locs': list of locations (upper-left-hand-corners, 1x2 ndarray) for ls_images
        'ls_strings': list of strings specifying the output image file names
        'thumbnail_width': desired width of the thumbnanil
        'thumbnail_height': desired height of the thumbnanil
        """
        
        if dict_in.has_key('thumbnail_columns'):
            self.thumbnail_columns = dict_in['thumbnail_columns']
        if dict_in.has_key('thumbnail_rows'):
            self.thumbnail_rows = dict_in['thumbnail_rows']
        if dict_in.has_key('thumbnail_width'):
            self.thumbnail_width = dict_in['thumbnail_width']
        if dict_in.has_key('thumbnail_height'):
            self.thumbnail_height = dict_in['thumbnail_height']
        #create the output directory if it doesn't exist
        if not os.path.exists(self.output_directory):
            os.mkdir(self.output_directory)

        thumbnails = dict_in['ls_images']
        thumbnail_filenames = dict_in['ls_strings']
        thumbnail_locs = dict_in['ls_locs']

        MONTAGE_IMG_ALT      = 'montage'
        MONTAGE_MAP_ID       = 'thumbnailmontage'
        MONTAGE_JS           =  os.path.realpath(__file__) + 'cmap.js'
        THUMBNAIL_IMG_ID     = 'thumbnailimage'

        montage_url    = self.output_directory + '/' + self.montage_im_file
        image_basename = self.output_directory

        ######################################################################
        montage_width  = (self.thumbnail_width * self.thumbnail_columns)
        montage_height = (self.thumbnail_height * self.thumbnail_rows)
        montage_img    = Image.new('RGB', (montage_width, montage_height))
        montage_draw   = ImageDraw.Draw(montage_img)

        img_tag  = '<img src="%s" width="%d" height="%d" alt="%s" border="0" usemap="#%s" />\n'
        map_tag  = '<map id="%s" name="%s">\n'
        area_tag = '  <area shape="rect" '         \
                           'coords="%d,%d,%d,%d" ' \
                           'href="%s" '            \
                           'onmouseover="javascript:cmap_show_image(\'%s\', \'%s\', %d, \'%s\');"/>\n'

        # Create html file.
        strPath=self.output_directory + '/' + self.montage_html_file
        html_output = open(strPath, 'w')

        # Insert javascript.
        html_output.write('<script language="JavaScript">\n');
        html_output.write(open(MONTAGE_JS).read() % (len(thumbnails), image_basename))
        html_output.write('</script>\n\n');

        # Add image tag for thumbnail montage.
        html_output.write('<table border="0"><tr><td>\n')
        html_output.write(img_tag % (montage_url, montage_width, montage_height,
                                     MONTAGE_IMG_ALT, MONTAGE_MAP_ID))
        html_output.write(map_tag % (MONTAGE_MAP_ID, MONTAGE_MAP_ID))

        for ix, thumbnail in enumerate(thumbnails):
            inum = ix + 1
            #save the normalized thumbnail
            strPath=self.output_directory + '/' + str(thumbnail_filenames[ix])
            f = open(strPath,'wb')
            w = png.Writer(*(thumbnail.shape[1],thumbnail.shape[0]),greyscale=(thumbnail.ndim==2))
            thumbnail[thumbnail<0]=0
            w.write(f,thumbnail/np.max(thumbnail)*255)
            f.close()

            #load thumbnail image, resize it and paste it into the montage.
            img   = Image.open(strPath)
            img   = img.resize((self.thumbnail_width, self.thumbnail_height))
            xpos = thumbnail_locs[ix][0]
            ypos = thumbnail_locs[ix][1]
            xpos2 = xpos + self.thumbnail_width
            ypos2 = ypos + self.thumbnail_height
            box = (xpos, ypos, xpos2, ypos2)
            montage_img.paste(img, box)

            # Add area tag for this image to the map.
            ititle = thumbnail_filenames[ix]
            ilink = thumbnail_filenames[ix]
            html_output.write(area_tag % (xpos, ypos, xpos2, ypos2,
                                          ilink,
                                          THUMBNAIL_IMG_ID, MONTAGE_MAP_ID, inum, ititle))

        # montage_img.save(self.montage_im_file, 'JPEG')
        montage_img.save(montage_url, 'JPEG')

        # Finish html.
        html_output.write('</map>\n')

        html_output.write('</td>\n')
        html_output.write('<td valign="center">\n')
        img_tag = '<img src="%s%03d.png" id="%s"  border="0" alt="%s" />\n'
        html_output.write(img_tag % (image_basename, len(thumbnails) , THUMBNAIL_IMG_ID, MONTAGE_IMG_ALT))
        html_output.write('</tr></table>\n')

        html_output.close()
            
    class Factory:
        def create(self,ps_parameters,str_section):
            return OutputMontage(ps_parameters,str_section)
