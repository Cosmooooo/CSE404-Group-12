% Write data to csv file from matlab file

folder = "ytcelebrity_init/Init_State/";
files = dir(fullfile(folder, '*.mat'));
cell = {};
for i = 1:size(files, 1)
    file_name = files(i).name;
    full_name = folder + file_name;
    state = load(full_name).gp;
    cell{end + 1, 1} = file_name(1:end-4) + ".avi";
    state = state';
    for j = 1:size(state,2)
        cell{end, j + 1} = state(j);
    end
end

T = cell2table(cell, "VariableNames",["filename","pos-x", "pos-y", "scale", "rot", "*","**"]);
writetable(T,'../data.csv');