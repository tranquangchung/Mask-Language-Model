#CUDA_VISIBLE_DEVICES=0 python __main__.py -c ./data/test_1w.txt -t ./data/test_1.txt -v ./data/char_sort.vocab -o ./output_1
#CUDA_VISIBLE_DEVICES=0 python __main__1.py -c ./data/test_1w.txt -t ./data/test_1.txt -v ./data/char_sort.vocab -o ./output_tts
#CUDA_VISIBLE_DEVICES=0 python __main__2.py -c ./data/test_1w.txt -t ./data/test_1.txt -v ./data/char_sort.vocab -o ./output_ttsAdap
#CUDA_VISIBLE_DEVICES=0 python __main__3.py -c ./data/test_1w.txt -t ./data/test_1.txt -v ./data/char_sort.vocab -o ./output_ttsstyle


#python valid_model.py -m ./output_tts/model_mlm/mlm_ep19.model -v ./data/char_sort.vocab
python valid_modeltts.py -m ./output_ttsAdap/model_mlm/mlm_ep99.model -v ./data/char_sort.vocab