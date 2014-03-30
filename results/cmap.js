// The percent format fields are filled in by montage_writer.py
var cmap_n_images        = %d;
var cmap_image_base      = '%s';
var cmap_image_urls      = Array();
var cmap_image_objs      = Array();

function cmap_show_image(imgid, mapid, inum, ititle)
{
	var img = document.getElementById(imgid);
	var map = document.getElementById(mapid);
	var i;
	var j;

	if ( img  &&  map ) {
		if ( cmap_image_urls.length == 0 ) {
			// Create image urls and image objects.
			for ( i = 0; i < cmap_n_images; i++ ) {
				j = i + 1;
				if      ( j < 10  ) zero = '00';
				else if ( j < 100 ) zero = '0';
				else                zero = '';
				cmap_image_urls[i] = cmap_image_base + zero + j + '.png';
				cmap_image_objs[i] = new Image();
			}
		}

		i = inum - 1;
		if ( !cmap_image_objs[i].src ) {
			cmap_image_objs[i].src = cmap_image_urls[i];
		}
		img.src   = cmap_image_objs[i].src;
		map.title = ititle
	}
}
