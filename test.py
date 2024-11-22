import torch
import numpy as np
fingerprint = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]) #embedded fingerprint
all_rand_fin = []
def generate_random_fingerprints(fingerprint_length, batch_size=4, size=(400, 400)):
    #2 excluded, it creates a tensor of 0 and 1 with batch_size x fingerprint_size size
    #use the following as default (original code)
    #z = torch.zeros((batch_size, fingerprint_length), dtype=torch.float).random_(0, 2)

    #use the following three lines of code to minimize the randomness
    #I use a seed to make the pseudo-random sequence generation the same for every batch
    #torch.manual_seed(42)
    #if torch.cuda.is_available():
        #torch.cuda.manual_seed(42)

    z = torch.zeros((batch_size, fingerprint_length), dtype=torch.float).random_(0, 2)
    all_rand_fin.append(z)

    return z

def main():


    for i in range(20):
        for j in range(2444):
            fingerprints = generate_random_fingerprints(
                100, 64, (128,128)
            )


    print("numero di tutte le firme")
    print(len(all_rand_fin))
    #print(all_rand_fin)
   
   
    """
    ok_check = False
    seed = 32567894
    while(ok_check==False):
        print("Primo while")
        seed += 1
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        z = torch.zeros((1, 100), dtype=torch.float).random_(0, 2)


        for zi in all_rand_fin:
            for fin in zi:
                #print(fin)
                #print(zi)
                if torch.equal(fin, z):
                    print("Seed in fin")
                    print(seed)
                    ok_check = True
    """

    


    


    i = -1
    
    ok_check=False
    for fins_64 in all_rand_fin:
        for fin in fins_64:
            i+=1
            if i==0:  
                torch.set_printoptions(precision=8, sci_mode=False)
                print(fin)
                        
            if torch.equal(fin,fingerprint):
                ok_check = True  
    if ok_check == False:
        print("fingerprint is not in fin")
   
   


if __name__ == "__main__":
    main()
