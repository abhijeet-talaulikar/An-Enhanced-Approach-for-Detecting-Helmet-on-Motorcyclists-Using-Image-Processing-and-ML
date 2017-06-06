<?php
include 'ImageResize.php';
use \Eventviva\ImageResize;
$dir = new DirectoryIterator(dirname(__FILE__));
$i = 0;
foreach ($dir as $fileinfo) {
    if (!$fileinfo->isDot()) {
	$file = explode('.', $fileinfo->getFilename());
	$ext = $file[count($file)-1];
	if($ext != 'php' && $ext != 'php~') {
		$name = $fileinfo->getFilename();
		try {
			$image = new ImageResize($name);
			$image->resize(100, 100, $allow_enlarge = True);
			unlink($name);
			$image->save($name);
		} catch(Exception $e) {}
	}
    }
}
?>
