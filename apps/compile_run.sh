#set -x

APP=${1%/}

if [ -z $APP ]; then
	echo "<script> app_folder_name"
	echo "Example:"
	echo "<script> bilateral_grid"
else
	cd ../
	pwd
	WITH_MPI=1 make -j3
	cd -
	cd $APP
	WITH_MPI=1 make -j3 -B distributed_${APP}
	HL_DEBUG_CODEGEN=1 ./distributed_${APP} 512 512 512
fi
