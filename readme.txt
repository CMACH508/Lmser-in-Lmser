��ĿĿ¼�����
Lmser_in_Lmser......
    ......data                       ###���ݼ�����txt�ļ����Ŀ¼
    ......model                      ###ģ��ʵ���ļ����Ŀ¼
    ......NLDA                       ###matlab��������NLDAָ����ļ�Ŀ¼����Ҫ�����ڷ���������matlab������ã�������Ҫ���úò��Եķ���
    ......pytorch_fid                ###FIDָ��pytorchʵ���ļ����Ŀ¼������Ҫ�������У����þͿ���
    ......util                       ###����Ԥ����ʵ�ַ�������Ŀ¼
    FeatureSIM.m                     ###����FSIMָ��ʵ�ַ����ļ�������Ҫ�������У�test_Method.py����
    option.py                        ###���������ļ�###
    test_Method.py                   ###����FID��FSIMָ�귽��
    test_lmser_in_lmser.py           ###ģ�Ͳ����ļ������ݲ��Լ�������Ƭ��������ͼƬ������test_Method.py���Ծ���ָ��
    train_lmser_in_lmser.py          ###ģ��ѵ���ļ�


������������
ѵ����
    source activate py36             ###����conda�е�py36����
    tmux �� tmux a                   ###��tmux�ն˸��������ں�̨ѵ��ģ��
    python train_lmser_in_lmser.py --gpu "0,1" --data_type CUFS --sub_node lmserinlmser_cufs
                                     ###����ѵ��ģ�ͽ��̣���������Ҫ������gpu�����ݼ����ͣ�ģ�ͱ���·��(������ʱ�ķ�����)
                                     ###����ɼ�option.py�ļ�
    python train_lmser_in_lmser.py --gpu "0,1" --data_type CUFS --sub_node lmserinlmser_cufs --resume_training 1
                                     ###����ѵ�����̣�����API���⣬resume_training��bool���Ͳ�����Ҫ����Ĭ��ֵΪfalse
                                     ###����Ҫ����Ϊtrueʱ��ֻҪ��--resume_training + ���һ��ֵ�Ϳ���
���ԣ�
    python test_lmser_in_lmser.py --gpu XX --sub_node XX --data_type XX
                                     ###ֻ�������ֲ�������option.pyû�й�ϵ��ͨ������IDE���м��ɣ�����Ҫ�ڷ�������������������
    python test_Method.py            ###����IDE����
    /data/shengqingjie/matlab/bin/matlab -nodisplay -r NLDA_multiple.m
    /data/shengqingjie/matlab/bin/matlab -nodisplay -r NLDA_single.m
                                     ###����NLDAָ�꣬��Ҫ��cd��NLDA_multiple.m��NLDA_single.m����Ŀ¼��������tmux�ն�����

