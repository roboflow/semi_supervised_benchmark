python dispatcher.py stac.py url_list.txt --model_name=yolo11n --skip_stac=True --batch=1 &
python dispatcher.py stac.py url_list.txt --model_name=yolo11n --skip_stac=True --batch=2 &
python dispatcher.py stac.py url_list.txt --model_name=yolo11m --skip_stac=True --batch=128 &
wait

python dispatcher.py stac.py url_list.txt --model_name=yolo11s --skip_stac=True --batch=1 &
python dispatcher.py stac.py url_list.txt --model_name=yolo11s --skip_stac=True --batch=2 &
python dispatcher.py stac.py url_list.txt --model_name=yolo11m --skip_stac=True --batch=64 &
wait

python dispatcher.py stac.py url_list.txt --model_name=yolo11m --skip_stac=True --batch=1 &
python dispatcher.py stac.py url_list.txt --model_name=yolo11m --skip_stac=True --batch=16 &
python dispatcher.py stac.py url_list.txt --model_name=yolo11m --skip_stac=True --batch=32 &
wait

python dispatcher.py stac.py url_list.txt --model_name=yolo11m --skip_stac=True --batch=2 &
python dispatcher.py stac.py url_list.txt --model_name=yolo11n --skip_stac=True --batch=4 &
python dispatcher.py stac.py url_list.txt --model_name=yolo11s --skip_stac=True --batch=128 &
wait

python dispatcher.py stac.py url_list.txt --model_name=yolo11m --skip_stac=True --batch=4 &
python dispatcher.py stac.py url_list.txt --model_name=yolo11s --skip_stac=True --batch=4 &
python dispatcher.py stac.py url_list.txt --model_name=yolo11n --skip_stac=True --batch=128 &
wait

python dispatcher.py stac.py url_list.txt --model_name=yolo11s --skip_stac=True --batch=32 &
python dispatcher.py stac.py url_list.txt --model_name=yolo11n --skip_stac=True --batch=32 &
python dispatcher.py stac.py url_list.txt --model_name=yolo11s --skip_stac=True --batch=8 &
wait

python dispatcher.py stac.py url_list.txt --model_name=yolo11m --skip_stac=True --batch=8 &
python dispatcher.py stac.py url_list.txt --model_name=yolo11s --skip_stac=True --batch=16 &
python dispatcher.py stac.py url_list.txt --model_name=yolo11n --skip_stac=True --batch=64 &
wait

python dispatcher.py stac.py url_list.txt --model_name=yolo11n --skip_stac=True --batch=16 &
python dispatcher.py stac.py url_list.txt --model_name=yolo11s --skip_stac=True --batch=64 &
python dispatcher.py stac.py url_list.txt --model_name=yolo11n --skip_stac=True --batch=8 &
wait
