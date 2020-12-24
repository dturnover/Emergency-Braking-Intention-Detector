dinfo = dir('~/Desktop/CS470/Final_Project/Emergency_Braking_EEG_Data');
names_cell = {dinfo.name};
names_cell = names_cell(:,3:end);

N = 8;
data = cell(N,1);
for k = 1:N
    data{k} = load(names_cell{k});
end

% initialize important variables in regards to dividing data into training and testing, as well as downsampling, and what your eeg signals are
ttSplit = .75; % the ratio which divides subjects into training or testing
all_features = {1:N};
numSkips = 10; % increase to downsample, decrease to upsample
numEEG = 62;

for i = 1:N
    % initialize important variables in regards to sampling rate, and size of your data
    cntSize = numel(data{i,1}.cnt.x(:,1)); 
    numEvents = numel(data{i,1}.mrk.time);
    ad_size = round(cntSize/numSkips);
    altered_data = zeros(ad_size, 65);
    srd = 5; % divisible amount to account for sample rate. For a rate of 200 it would be 5
    adjustedTime = data{i,1}.mrk.time / srd; %the eeg data is 1 sample every 5 miliseconds. Mrk.time is in miliseconds
    
    % preprocess data
    ppSub = mean(data{i,1}.cnt.x(1:100/srd, 1:numEEG)); % must subtract average of first 100 ms worth of samples
    for pp = 101:cntSize
        data{i,1}.cnt.x(pp, 1:numEEG) = data{i,1}.cnt.x(pp, 1:numEEG) - ppSub;
    end
    
    % extract only the eeg data as well as downsample
    itr = 1;
    for fill = 1:numSkips:cntSize
         altered_data(itr,1:numEEG) = data{i,1}.cnt.x(fill,1:numEEG); % the eeg data
         altered_data(itr,numEEG+1) = numSkips*srd*itr; % the time in miliseconds (5 miliseconds per sample, skipping every numSkips samples)
         itr = itr + 1;
    end
    
    % add label to corresponding raw data
    for j = 1:numEvents
       if data{i,1}.mrk.y(5,j) == 1 % 5 indicates we're finding the target based off react_emg, 2 indicates we're finding the target based on the lead care braking
           approx_index = round(adjustedTime(j)/numSkips); 
           for k = approx_index-round((1300/srd)/numSkips):approx_index+round((200/srd)/numSkips) % The target duration at the time of react_emg is -1300 ms and +200 ms or -300 ms and + 1200
                altered_data(k,numEEG+2) = 1; %label as braking
           end
       end
    end
    
    % extract features for training and testing
    targetSize = round((1500/srd)/numSkips); % 1500 miliseconds, divided by srd, divided by downsampling amount
    numCategories = 5;
    numFR = numCategories*numEEG+1;
    features = zeros(round(ad_size/targetSize), numFR);

    itr = 1;
    m = 1;
    while m < ad_size-3*targetSize
        spaceCheck = m+round((3000/srd)/numSkips); % looks 3000 miliseconds ahead to make sure there are no braking events ahead

        if (altered_data(spaceCheck,numEEG+2) == 1) % there is a braking event coming up
            m = m+round((3000/srd)/numSkips); % skip to the known braking point
            % label
            features(itr,numFR) = 1;

            % features
            ADwindow = altered_data(m:m+targetSize,1:numEEG);
            % mean
            features(itr,1:numEEG) = mean(ADwindow);      
            % standard deviation
            features(itr,numEEG+1:2*numEEG) = std(ADwindow); 
            % activity
            features(itr,2*numEEG+1:3*numEEG) = var(ADwindow); 
            % mobility and complexity
            for d = 1:numEEG
                ADwd0 = ADwindow(:,d);
                ADwd1 = zeros(1,length(ADwd0));
                ADwd2 = zeros(1,length(ADwd0));
                for d1 = 2:length(ADwd0)
                    ADwd1(d1) = ADwd0(d1) - ADwd0(d1-1);
                end
                % mobility
                features(itr,3*numEEG+d) = std(ADwd1)/std(ADwd0); 
                for d2=3:length(ADwd0)
                    ADwd2(d2) = ADwd0(d2) - 2*ADwd0(d2-1) + ADwd0(d2-2);
                end
                FF = (std(ADwd2)/std(ADwd1))/(std(ADwd1)/std(ADwd0));
                % complexity
                features(itr,4*numEEG+d) = FF;
            end              
            m = m+1 + targetSize + round((3000/srd)/numSkips); % move end not only to then end of the target but also 3000ms ahead of it

        else % this is a non braking event
            % label
            features(itr,numFR) = 0;

            % features 
            ADwindow = altered_data(m:m+targetSize,1:numEEG);
            % mean
            features(itr,1:numEEG) = mean(ADwindow);       
            % standard deviation
            features(itr,numEEG+1:2*numEEG) = std(ADwindow); 
            % activity
            features(itr,2*numEEG+1:3*numEEG) = var(ADwindow); 
            % mobility and complexity
            for d = 1:numEEG
                ADwd0 = ADwindow(:,d);
                ADwd1 = zeros(1,length(ADwd0));
                ADwd2 = zeros(1,length(ADwd0));
                for d1 = 2:length(ADwd0)
                    ADwd1(d1) = ADwd0(d1) - ADwd0(d1-1);
                end
                % mobility
                features(itr,3*numEEG+d) = std(ADwd1)/std(ADwd0); 
                for d2=3:length(ADwd0)
                    ADwd2(d2) = ADwd0(d2) - 2*ADwd0(d2-1) + ADwd0(d2-2);
                end
                FF = (std(ADwd2)/std(ADwd1))/(std(ADwd1)/std(ADwd0));
                % complexity
                features(itr,4*numEEG+d) = FF;
            end              
            m = m+1 + targetSize;
        end            
        itr = itr + 1;
    end       
    all_features{i} = features;
end

% divide into training and testing data and convert cell matrices to tables
converge_training = all_features{1}; % will concatenate all training data into a single cell matrix
for train_split = 2:round(N*ttSplit) % N/2 assumes we split training and testing in half
    converge_training = cat(1, converge_training, all_features{train_split});
end

converge_testing = all_features{round(N*ttSplit)+1};% round(N*ttSplit)+1};
for test_split = round(N*ttSplit)+2:N
     converge_testing = cat(1, converge_testing, all_features{test_split}); 
end

Training = array2table(converge_training, 'VariableNames', {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100','101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120','121','122','123','124','125','126','127','128','129','130','131','132','133','134','135','136','137','138','139','140','141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160','161','162','163','164','165','166','167','168','169','170','171','172','173','174','175','176','177','178','179','180','181','182','183','184','185','186','187','188','189','190','191','192','193','194','195','196','197','198','199','200','201','202','203','204','205','206','207','208','209','210','211','212','213','214','215','216','217','218','219','220','221','222','223','224','225','226','227','228','229','230','231','232','233','234','235','236','237','238','239','240','241','242','243','244','245','246','247','248','249','250','251','252','253','254','255','256','257','258','259','260','261','262','263','264','265','266','267','268','269','270','271','272','273','274','275','276','277','278','279','280','281','282','283','284','285','286','287','288','289','290','291','292','293','294','295','296','297','298','299','300','301','302','303','304','305','306','307','308','309','310','Label'});
Testing = array2table(converge_testing, 'VariableNames', {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100','101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120','121','122','123','124','125','126','127','128','129','130','131','132','133','134','135','136','137','138','139','140','141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160','161','162','163','164','165','166','167','168','169','170','171','172','173','174','175','176','177','178','179','180','181','182','183','184','185','186','187','188','189','190','191','192','193','194','195','196','197','198','199','200','201','202','203','204','205','206','207','208','209','210','211','212','213','214','215','216','217','218','219','220','221','222','223','224','225','226','227','228','229','230','231','232','233','234','235','236','237','238','239','240','241','242','243','244','245','246','247','248','249','250','251','252','253','254','255','256','257','258','259','260','261','262','263','264','265','266','267','268','269','270','271','272','273','274','275','276','277','278','279','280','281','282','283','284','285','286','287','288','289','290','291','292','293','294','295','296','297','298','299','300','301','302','303','304','305','306','307','308','309','310','Label'});

% train the support vector machine
model = fitcsvm(Training,'Label','Standardize',true,'KernelFunction','rbf','OptimizeHyperparameters','auto');

% use the trained model to make predictions on the testing data
predicted_labels = predict(model,Testing);

% see how well the model did
actual_labels = converge_testing(:,numCategories*numEEG+1); 

numCorrect = predicted_labels == actual_labels;
percent_correct = sum(numCorrect)/numel(numCorrect);
percent_incorrect = 1 - percent_correct;

% confusion matrix
cf = confusionchart(actual_labels, predicted_labels)
cf.ColumnSummary = 'column-normalized';
cf.RowSummary = 'row-normalized';
Matrix = cf.NormalizedValues; 

true_positives = Matrix(2,2); 
true_negatives = Matrix(1,1);
false_positives = Matrix(1,2);  
false_negatives = Matrix(2,1);

precision = true_positives/(true_positives + false_positives) 
recall = true_positives/(true_positives + false_negatives)
f1_score = (2 * (precision * recall))/(precision + recall)