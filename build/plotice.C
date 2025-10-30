// -------------------------------------------------------------------
// -------------------------------------------------------------------
//
// *********************************************************************
// To execute this macro under ROOT after your simulation ended,
//   1 - launch ROOT (usually type 'root' at your machine's prompt)
//   2 - type '.X plot.C' at the ROOT session prompt
// *********************************************************************

void SetLeafAddress(TNtuple* ntuple, const char* name, void* address);

void plotice()
{
  gROOT->Reset();
  gStyle->SetPalette(1);
  gROOT->SetStyle("Plain");
  gStyle->SetOptTitle(0);

  TCanvas* c1 = new TCanvas ("c1","",20,20,2500,500);
  c1->Divide(5,1);

  // Uncomment if merging should be done
  //system ("rm -rf dna.root");
  //system ("hadd dna.root dna_*.root");

  TFile* f = new TFile("dna.root");

  TNtuple* ntuple;
  ntuple = (TNtuple*)f->Get("step");
  bool rowWise = true;
  TBranch* eventBranch = ntuple->FindBranch("row_wise_branch");
  if ( ! eventBranch ) rowWise = false;
  // std::cout <<  "rowWise: " << rowWise << std::endl;

  //*********************************************************************
  // canvas tab 1
  //*********************************************************************

  c1->cd(1);
  gStyle->SetOptStat(000000);

  // All
  ntuple->SetFillStyle(1001);
  ntuple->SetFillColor(2);
  ntuple->Draw("flagProcess","","B");

  // Excitation
  ntuple->SetFillStyle(1001);
  ntuple->SetFillColor(3);
  ntuple->Draw("flagProcess","flagProcess==12||flagProcess==15||flagProcess==22||flagProcess==32||flagProcess==42||flagProcess==52||flagProcess==62","Bsame");

  // Elastic
  ntuple->SetFillStyle(1001);
  ntuple->SetFillColor(4);
  ntuple->Draw("flagProcess","flagProcess==11||flagProcess==21||flagProcess==31||flagProcess==41||flagProcess==51||flagProcess==61||flagProcess==110||flagProcess==210||flagProcess==410||flagProcess==510||flagProcess==710||flagProcess==120||flagProcess==220||flagProcess==420||flagProcess==520||flagProcess==720","Bsame");

  // Ionisation
  ntuple->SetFillStyle(1001);
  ntuple->SetFillColor(5);
  ntuple->Draw("flagProcess","flagProcess==13||flagProcess==23||flagProcess==33||flagProcess==43||flagProcess==53||flagProcess==63||flagProcess==73||flagProcess==130||flagProcess==230||flagProcess==430||flagProcess==530||flagProcess==730","Bsame");

  // Charge decrease
  //ntuple->SetFillStyle(1001);
  //ntuple->SetFillColor(6);
  //ntuple->Draw("flagProcess","flagProcess==24||flagProcess==44||flagProcess==54","Bsame");

  // Charge increase
  //ntuple->SetFillStyle(1001);
  //ntuple->SetFillColor(7);
  //ntuple->Draw("flagProcess","flagProcess==35||flagProcess==55||flagProcess==65","Bsame");

  gPad->SetLogy();

  //*********************************************************************
  // canvas tab 2
  //*********************************************************************

  c1->cd(2);

  ntuple->SetMarkerColor(2);

  ntuple->Draw("x:y:z","flagParticle==1");

  //ntuple->SetMarkerColor(4);
  //ntuple->SetMarkerSize(4);
  //ntuple->Draw("x:y:z/1000","flagParticle==4 || flagParticle==5 || flagParticle==6","same");

  //*********************************************************************
  // canvas tab 3
  //*********************************************************************

  c1->cd(3);

  Double_t flagParticle;
  Double_t flagProcess;
  Double_t x;
  Double_t y;
  Double_t z;
  Double_t totalEnergyDeposit;
  Double_t stepLength;
  Double_t kineticEnergyDifference;
  Int_t eventID;
  Double_t kineticEnergy;
  Int_t stepID;
  Int_t trackID;
  Int_t parentID;
  Double_t angle;

  if ( ! rowWise ) {
    ntuple->SetBranchAddress("flagParticle",&flagParticle);
    ntuple->SetBranchAddress("flagProcess",&flagProcess);
    ntuple->SetBranchAddress("x",&x);
    ntuple->SetBranchAddress("y",&y);
    ntuple->SetBranchAddress("z",&z);
    ntuple->SetBranchAddress("totalEnergyDeposit",&totalEnergyDeposit);
    ntuple->SetBranchAddress("stepLength",&stepLength);
    ntuple->SetBranchAddress("kineticEnergyDifference",&kineticEnergyDifference);
    ntuple->SetBranchAddress("kineticEnergy",&kineticEnergy);
    ntuple->SetBranchAddress("cosTheta",&angle);
    ntuple->SetBranchAddress("eventID",&eventID);
    ntuple->SetBranchAddress("trackID",&trackID);
    ntuple->SetBranchAddress("parentID",&parentID);
    ntuple->SetBranchAddress("stepID",&stepID);
  }
  else {
    SetLeafAddress(ntuple, "flagParticle",&flagParticle);
    SetLeafAddress(ntuple, "flagProcess",&flagProcess);
    SetLeafAddress(ntuple, "x",&x);
    SetLeafAddress(ntuple, "y",&y);
    SetLeafAddress(ntuple, "z",&z);
    SetLeafAddress(ntuple, "totalEnergyDeposit",&totalEnergyDeposit);
    SetLeafAddress(ntuple, "stepLength",&stepLength);
    SetLeafAddress(ntuple, "kineticEnergyDifference",&kineticEnergyDifference);
    SetLeafAddress(ntuple, "kineticEnergy",&kineticEnergy);
    SetLeafAddress(ntuple, "cosTheta",&angle);
    SetLeafAddress(ntuple, "eventID",&eventID);
    SetLeafAddress(ntuple, "trackID",&trackID);
    SetLeafAddress(ntuple, "parentID",&parentID);
    SetLeafAddress(ntuple, "stepID",&stepID);
  }

  TH1F* hsolvE = new TH1F ("hsolvE","solvE",100,0,2000);
  TH1F* helastE = new TH1F ("helastE","elastE",100,0,2000);
  TH1F* hexcitE = new TH1F ("hexcitE","excitE",100,0,2000);
  TH1F* hioniE = new TH1F ("hiioniE","ioniE",100,0,2000);
  TH1F* hattE = new TH1F ("hattE","attE",100,0,2000);
  TH1F* hvibE = new TH1F ("hvibE","vibE",100,0,2000);

  for (Int_t j=0;j<ntuple->GetEntries(); j++)
  {
    ntuple->GetEntry(j);
    if (flagProcess==10) hsolvE->Fill(x);
    if (flagProcess==11) helastE->Fill(x);
    if (flagProcess==12) hexcitE->Fill(x);
    if (flagProcess==13) hioniE->Fill(x);
    if (flagProcess==14) hattE->Fill(x);
    if (flagProcess==15) hvibE->Fill(x);

  }

  helastE->GetXaxis()->SetTitle("x (nm)");
  helastE->SetLineColor(2);

  hexcitE->SetLineColor(3);
  hioniE->SetLineColor(4);
  hattE->SetLineColor(5);
  hvibE->SetLineColor(6);
  hsolvE->SetLineColor(7);

  gPad->SetLogy();

  helastE->Draw("");
  hexcitE->Draw("SAME");
  hioniE->Draw("SAME");
  hattE->Draw("SAME");
  hvibE->Draw("SAME");
  hsolvE->Draw("SAME");

  //*********************************************************************
  // canvas tab 4
  //*********************************************************************

  TNtuple* ntuple2;
  //ntuple2 = (TNtuple*)f->Get("track");
  ntuple2 = (TNtuple*)f->Get("step");
  bool rowWise2 = true;
  TBranch* eventBranch2 = ntuple2->FindBranch("row_wise_branch");
  if ( ! eventBranch2 ) rowWise2 = false;

  c1->cd(4);
  gStyle->SetOptStat(000000);

  // All
  ntuple2->SetFillStyle(1001);
  ntuple2->SetFillColor(2);
  ntuple2->Draw("kineticEnergy","flagParticle==1","B");

  gPad->SetLogy();

  //*********************************************************************
  // canvas tab 5: Cross section vs kinetic energy (per channel)
  //*********************************************************************
  c1->cd(5);
  gStyle->SetOptStat(0);

  // We'll render per-channel plots on a dedicated canvas sized so that
  // each panel matches the size of other panels (~500x500). The tab here
  // will show a short note and summary only.
  const double nH2O_cm3 = 3.343e22; // water molecules per cm^3
  TLeaf* leafCI = ntuple->FindLeaf("channelIndex");
  int channels[256]; int nChan = 0;
  if (leafCI) {
    TH1I* hci = new TH1I("hci","channelIndex occupancy",256,-0.5,255.5);
    ntuple->Draw("channelIndex>>hci","channelIndex>=0","goff");
    for (int b=1; b<=hci->GetNbinsX(); ++b) {
      if (hci->GetBinContent(b) > 0) {
        int k = (int)std::round(hci->GetBinCenter(b));
        if (k >= 0 && k < 256) { channels[nChan++] = k; }
      }
    }
  }

  // Show a brief note in the tab
  gPad->Clear();
  TLatex note; note.SetNDC(); note.SetTextSize(0.05);
  if (!leafCI) {
    note.DrawLatex(0.10, 0.60, "No channelIndex in file; drawing combined XS");
    ntuple->SetMarkerStyle(20);
    ntuple->SetMarkerSize(0.6);
    ntuple->SetMarkerColor(kBlue+1);
    ntuple->Draw(Form("(vibCrossSection*10/%g)*1e16:kineticEnergy", nH2O_cm3),
                 "vibCrossSection>0",
                 "COLZ");
    TH2F* htemp = (TH2F*)gPad->GetPrimitive("htemp");
    if (htemp) {
      htemp->GetXaxis()->SetTitle("Kinetic Energy (eV)");
      htemp->GetYaxis()->SetTitle("Cross Section (10^{-16} cm^{2})");
    }
  } else if (nChan == 0) {
    note.DrawLatex(0.10, 0.60, "No non-negative channelIndex entries; combined XS");
    ntuple->SetMarkerStyle(20);
    ntuple->SetMarkerSize(0.6);
    ntuple->SetMarkerColor(kBlue+1);
    ntuple->Draw(Form("(vibCrossSection*10/%g)*1e16:kineticEnergy", nH2O_cm3),
                 "vibCrossSection>0",
                 "COLZ");
    TH2F* htemp = (TH2F*)gPad->GetPrimitive("htemp");
    if (htemp) {
      htemp->GetXaxis()->SetTitle("Kinetic Energy (eV)");
      htemp->GetYaxis()->SetTitle("Cross Section (10^{-16} cm^{2})");
    }
  } else {
    // Build dedicated canvas for per-channel plots
    int ncols = std::min(5, nChan);
    int nrows = (nChan + ncols - 1) / ncols; // ceil
    int w = 500 * ncols;
    int h = 500 * nrows;
    TCanvas* cXS = new TCanvas("cXS", "Cross Section vs Energy by Channel", w, h);
    cXS->Divide(ncols, nrows);

    // Prepare black reference curves from data file (if available)
    std::vector<double> refE;
    std::vector<std::vector<double>> refXS(8);
    std::vector<TGraph*> refGraphs(8, nullptr);
    const char* led = gSystem->Getenv("G4LEDATA");
    if (led) {
      TString refPath = TString::Format("%s/dna/sigma_excitationvib_e_michaud.dat", led);
      std::ifstream fin(refPath.Data());
      if (fin) {
        while (true) {
          double e; if (!(fin >> e)) break;
          refE.push_back(e);
          for (int j=0;j<8;++j) {
            double v = 0.0; fin >> v; refXS[j].push_back(v); // in 1e-16 cm^2
          }
        }
        for (int j=0;j<8;++j) {
          if (!refE.empty() && refE.size() == refXS[j].size()) {
            refGraphs[j] = new TGraph((int)refE.size(), &refE[0], &refXS[j][0]);
            refGraphs[j]->SetLineColor(kBlack);
            refGraphs[j]->SetLineWidth(2);
          }
        }
      }
    }
    for (int i=0; i<nChan; ++i) {
      int k = channels[i];
      cXS->cd(i+1);
      ntuple->SetMarkerStyle(20);
      ntuple->SetMarkerSize(0.6);
      ntuple->SetMarkerColor(kBlue+1);
      if (ntuple->FindLeaf("channelMicroXS")) {
        ntuple->Draw("channelMicroXS*1e16:kineticEnergy",
                     Form("channelIndex==%d && channelMicroXS>0", k),
                     "COLZ");
      } else {
        ntuple->Draw(Form("(vibCrossSection*10/%g)*1e16:kineticEnergy", nH2O_cm3),
                     Form("channelIndex==%d && vibCrossSection>0", k),
                     "COLZ");
      }
      gPad->SetGrid();
      TH2F* htemp2 = (TH2F*)gPad->GetPrimitive("htemp");
      if (htemp2) {
        htemp2->GetXaxis()->SetTitle("Kinetic Energy (eV)");
        htemp2->GetYaxis()->SetTitle("Cross Section (10^{-16} cm^{2})");
        htemp2->SetTitle("");
      }
      // Overlay reference curve if available for this channel
      if (k >= 0 && k < (int)refGraphs.size() && refGraphs[k]) {
        refGraphs[k]->Draw("L SAME");
      }
      TLatex t; t.SetNDC(); t.SetTextSize(0.05);
      t.DrawLatex(0.15, 0.92, Form("Channel %d", k));
    }
    cXS->Modified();
    cXS->Update();

    // Note in the tab that a separate canvas was created
    note.DrawLatex(0.10, 0.60, "Opened 'cXS' canvas with per-channel panels");
  }

}

void SetLeafAddress(TNtuple* ntuple, const char* name, void* address) {
  TLeaf* leaf = ntuple->FindLeaf(name);
  if ( ! leaf ) {
    std::cerr << "Error in <SetLeafAddress>: unknown leaf --> " << name << std::endl;
    return;
  }
  leaf->SetAddress(address);
}
