Данный проект является моделью предсказания кредитного рейтинга банков-контрагентов исходя из их финансовых и бизнес данных за несколько месяцев. 

	Перед непосредственным обучением модели необходимо было загрузить данные из Excel файлов из папок 'Валидационная выборка' и 'banks'. Далее необходимо было сформировать датасет, сделать оверсеплинг, разделить на трейн и тест. Также было принято решение сделать биннинг бизнес показателей. 

	В ходе проекта было испробовано множество методов решения поставленной задачи, а именно:
1. Минимизация функции расстояния Кульбака-Лейбнера
	Техники: минимизация функции через scipy.minimize, ограничение коэффициентов модели.
	Файлы: 'preprocess_data_model.ipynb'

2. Минимизация функции потерь Triplet Loss
	Техники: преобразование датафрейма в нужную для функции форму, минимизация функции через scipy.minimize.
	Файлы: 'preprocess_data_model.ipynb'

3. Recurrent Neural Networks
	Техники: преобразование датафрейма в нужную для RNN форму ('сосиски' для каждого контрагента на 4 даты), использование PyTorch и CUDA для ускорения обучения модели (файл 'RNN.py'), использование оптимизатора RMSprop и планировщика scheduler для управления скоростью обучения, использование torch.tensor, организация цикла обучения, использование классов DataLoader и TensorDataset для эффективной обработки батчей данных во время обучения и валидации.
	Файлы: 'Only RNN.ipynb', 'RNN.py', преобразованные данные 'X_train_quad', 'X_test_quad', 'y_train_quad', 'y_test_quad'

4. Решающее дерево
	Техники: использование решающего дерева в задачи классификации, выявление и отбор значимых параметров, стресс тестирование модели.
	Файлы: 'KohenKappa_clear.ipynb', иллюстрация дерева 'tree1.pdf'

В файлах .ipynb приведены результаты тестирования модели.
