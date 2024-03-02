for filename in Tax-H36m-coco40k-Muco-UP-Mpii.tar human3.6m.tar coco_smpl.tar muco.tar up3d.tar mpii.tar 3dpw.tar
do
    ${azcopy_path}/azcopy copy "https://datarelease.blob.core.windows.net/metro/datasets/${filename}" ./
    tar xvf ${filename}
    rm ${filename}
done